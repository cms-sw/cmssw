/*
Track Quality Body file

C.Brown & C.Savard 07/2020
*/

#include "L1Trigger/TrackTrigger/interface/L1TrackQuality.h"
#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"

//Constructors

L1TrackQuality::L1TrackQuality() {}

L1TrackQuality::L1TrackQuality(const edm::ParameterSet& qualityParams) {
  std::string AlgorithmString = qualityParams.getParameter<std::string>("qualityAlgorithm");
  // Unpacks EDM parameter set itself to save unecessary processing within TrackProducers
  if (AlgorithmString == "Cut") {
    setCutParameters(AlgorithmString,
                     (float)qualityParams.getParameter<double>("maxZ0"),
                     (float)qualityParams.getParameter<double>("maxEta"),
                     (float)qualityParams.getParameter<double>("chi2dofMax"),
                     (float)qualityParams.getParameter<double>("bendchi2Max"),
                     (float)qualityParams.getParameter<double>("minPt"),
                     qualityParams.getParameter<int>("nStubsmin"));
  }

  else {
    setONNXModel(AlgorithmString,
                 qualityParams.getParameter<edm::FileInPath>("ONNXmodel"),
                 qualityParams.getParameter<std::string>("ONNXInputName"),
                 qualityParams.getParameter<std::vector<std::string>>("featureNames"));
    ONNXInvRScaling_ = qualityParams.getParameter<double>("ONNXInvRScale");
  }
}

std::vector<float> L1TrackQuality::featureTransform(TTTrack<Ref_Phase2TrackerDigi_>& aTrack,
                                                    std::vector<std::string> const& featureNames) {
  // List input features for MVA in proper order below, the features options are
  // {"log_chi2","log_chi2rphi","log_chi2rz","log_bendchi2","nstubs","lay1_hits","lay2_hits",
  // "lay3_hits","lay4_hits","lay5_hits","lay6_hits","disk1_hits","disk2_hits","disk3_hits",
  // "disk4_hits","disk5_hits","rinv","tanl","z0","dtot","ltot","chi2","chi2rz","chi2rphi",
  // "bendchi2","pt","eta","nlaymiss_interior","phi","bendchi2_bin","chi2rz_bin","chi2rphi_bin"}

  std::vector<float> transformedFeatures;

  // Define feature map, filled as features are generated
  std::map<std::string, float> feature_map;

  // The following converts the 7 bit hitmask in the TTTrackword to an expected
  // 11 bit hitmask based on the eta of the track
  std::vector<int> hitpattern_binary = {0, 0, 0, 0, 0, 0, 0};
  std::vector<int> hitpattern_expanded_binary = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> eta_bins = {0.0, 0.2, 0.41, 0.62, 0.9, 1.26, 1.68, 2.08, 2.4};

  // Expected hitmap table, each row corresponds to an eta bin, each value corresponds to
  // the expected layer in the expanded hit pattern. The expanded hit pattern should be
  // 11 bits but contains a 12th element so this hitmap table is symmetric
  int hitmap[8][7] = {{0, 1, 2, 3, 4, 5, 11},
                      {0, 1, 2, 3, 4, 5, 11},
                      {0, 1, 2, 3, 4, 5, 11},
                      {0, 1, 2, 3, 4, 5, 11},
                      {0, 1, 2, 3, 4, 5, 11},
                      {0, 1, 2, 6, 7, 8, 9},
                      {0, 1, 7, 8, 9, 10, 11},
                      {0, 6, 7, 8, 9, 10, 11}};

  // iterate through bits of the hitpattern and compare to 1 filling the hitpattern binary vector
  int tmp_trk_hitpattern = aTrack.hitPattern();
  for (int i = 6; i >= 0; i--) {
    int k = tmp_trk_hitpattern >> i;
    if (k & 1)
      hitpattern_binary[i] = 1;
  }

  // calculate number of missed interior layers from hitpattern
  int nbits = floor(log2(tmp_trk_hitpattern)) + 1;
  int lay_i = 0;
  int tmp_trk_nlaymiss_interior = 0;
  bool seq = false;
  for (int i = 0; i < nbits; i++) {
    lay_i = ((1 << i) & tmp_trk_hitpattern) >> i;  //0 or 1 in ith bit (right to left)

    if (lay_i && !seq)
      seq = true;  //sequence starts when first 1 found
    if (!lay_i && seq)
      tmp_trk_nlaymiss_interior++;
  }

  float eta = abs(aTrack.eta());
  int eta_size = static_cast<int>(eta_bins.size());
  // First iterate through eta bins

  for (int j = 1; j < eta_size; j++) {
    if (eta < eta_bins[j] && eta >= eta_bins[j - 1])  // if track in eta bin
    {
      // Iterate through hitpattern binary
      for (int k = 0; k <= 6; k++)
        // Fill expanded binary entries using the expected hitmap table positions
        hitpattern_expanded_binary[hitmap[j - 1][k]] = hitpattern_binary[k];
      break;
    }
  }

  int tmp_trk_ltot = 0;
  //calculate number of layer hits
  for (int i = 0; i < 6; ++i) {
    tmp_trk_ltot += hitpattern_expanded_binary[i];
  }

  int tmp_trk_dtot = 0;
  //calculate number of disk hits
  for (int i = 6; i < 11; ++i) {
    tmp_trk_dtot += hitpattern_expanded_binary[i];
  }

  // bin bendchi2 variable (bins from https://twiki.cern.ch/twiki/bin/viewauth/CMS/HybridDataFormat#Fitted_Tracks_written_by_KalmanF)
  float tmp_trk_bendchi2 = aTrack.stubPtConsistency();
  std::array<float, 8> bendchi2_bins{{0, 0.5, 1.25, 2, 3, 5, 10, 50}};
  int n_bendchi2 = static_cast<int>(bendchi2_bins.size());
  float tmp_trk_bendchi2_bin = -1;
  for (int i = 0; i < (n_bendchi2 - 1); i++) {
    if (tmp_trk_bendchi2 >= bendchi2_bins[i] && tmp_trk_bendchi2 < bendchi2_bins[i + 1]) {
      tmp_trk_bendchi2_bin = i;
      break;
    }
  }
  if (tmp_trk_bendchi2_bin < 0)
    tmp_trk_bendchi2_bin = n_bendchi2;

  // bin chi2rphi variable (bins from https://twiki.cern.ch/twiki/bin/viewauth/CMS/HybridDataFormat#Fitted_Tracks_written_by_KalmanF)
  float tmp_trk_chi2rphi = aTrack.chi2XY();
  std::array<float, 16> chi2rphi_bins{{0, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 40, 100, 200, 500, 1000, 3000}};
  int n_chi2rphi = static_cast<int>(chi2rphi_bins.size());
  float tmp_trk_chi2rphi_bin = -1;
  for (int i = 0; i < (n_chi2rphi - 1); i++) {
    if (tmp_trk_chi2rphi >= chi2rphi_bins[i] && tmp_trk_chi2rphi < chi2rphi_bins[i + 1]) {
      tmp_trk_chi2rphi_bin = i;
      break;
    }
  }
  if (tmp_trk_chi2rphi_bin < 0)
    tmp_trk_chi2rphi_bin = n_chi2rphi;

  // bin chi2rz variable (bins from https://twiki.cern.ch/twiki/bin/viewauth/CMS/HybridDataFormat#Fitted_Tracks_written_by_KalmanF)
  float tmp_trk_chi2rz = aTrack.chi2Z();
  std::array<float, 16> chi2rz_bins{{0, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 40, 100, 200, 500, 1000, 3000}};
  int n_chi2rz = static_cast<int>(chi2rz_bins.size());
  float tmp_trk_chi2rz_bin = -1;
  for (int i = 0; i < (n_chi2rz - 1); i++) {
    if (tmp_trk_chi2rz >= chi2rz_bins[i] && tmp_trk_chi2rz < chi2rz_bins[i + 1]) {
      tmp_trk_chi2rz_bin = i;
      break;
    }
  }
  if (tmp_trk_chi2rz_bin < 0)
    tmp_trk_chi2rz_bin = n_chi2rz;

  // get the nstub
  std::vector<edm::Ref<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>, TTStub<Ref_Phase2TrackerDigi_>>> stubRefs =
      aTrack.getStubRefs();

  // fill the feature map
  feature_map["nstub"] = stubRefs.size();
  feature_map["rinv"] = ONNXInvRScaling_ * abs(aTrack.rInv());
  feature_map["tanl"] = abs(aTrack.tanL());
  feature_map["z0"] = aTrack.z0();
  feature_map["phi"] = aTrack.phi();
  feature_map["pt"] = aTrack.momentum().perp();
  feature_map["eta"] = aTrack.eta();

  float tmp_trk_chi2 = aTrack.chi2();
  feature_map["chi2"] = tmp_trk_chi2;
  feature_map["log_chi2"] = log(tmp_trk_chi2);

  feature_map["chi2rphi"] = tmp_trk_chi2rphi;
  feature_map["log_chi2rphi"] = log(tmp_trk_chi2rphi);

  feature_map["chi2rz"] = tmp_trk_chi2rz;
  feature_map["log_chi2rz"] = log(tmp_trk_chi2rz);

  feature_map["chi2rz"] = tmp_trk_chi2rz;
  feature_map["log_chi2rz"] = log(tmp_trk_chi2rz);

  feature_map["bendchi2"] = tmp_trk_bendchi2;
  feature_map["log_bendchi2"] = log(tmp_trk_bendchi2);

  feature_map["lay1_hits"] = float(hitpattern_expanded_binary[0]);
  feature_map["lay2_hits"] = float(hitpattern_expanded_binary[1]);
  feature_map["lay3_hits"] = float(hitpattern_expanded_binary[2]);
  feature_map["lay4_hits"] = float(hitpattern_expanded_binary[3]);
  feature_map["lay5_hits"] = float(hitpattern_expanded_binary[4]);
  feature_map["lay6_hits"] = float(hitpattern_expanded_binary[5]);
  feature_map["disk1_hits"] = float(hitpattern_expanded_binary[6]);
  feature_map["disk2_hits"] = float(hitpattern_expanded_binary[7]);
  feature_map["disk3_hits"] = float(hitpattern_expanded_binary[8]);
  feature_map["disk4_hits"] = float(hitpattern_expanded_binary[9]);
  feature_map["disk5_hits"] = float(hitpattern_expanded_binary[10]);

  feature_map["dtot"] = float(tmp_trk_dtot);
  feature_map["ltot"] = float(tmp_trk_ltot);

  feature_map["nlaymiss_interior"] = float(tmp_trk_nlaymiss_interior);
  feature_map["bendchi2_bin"] = tmp_trk_bendchi2_bin;
  feature_map["chi2rphi_bin"] = tmp_trk_chi2rphi_bin;
  feature_map["chi2rz_bin"] = tmp_trk_chi2rz_bin;

  // fill tensor with track params
  transformedFeatures.reserve(featureNames.size());
  for (const std::string& feature : featureNames)
    transformedFeatures.push_back(feature_map[feature]);

  return transformedFeatures;
}

void L1TrackQuality::setL1TrackQuality(TTTrack<Ref_Phase2TrackerDigi_>& aTrack) {
  if (this->qualityAlgorithm_ == QualityAlgorithm::Cut) {
    // Get Track parameters
    float trk_pt = aTrack.momentum().perp();
    float trk_bend_chi2 = aTrack.stubPtConsistency();
    float trk_z0 = aTrack.z0();
    float trk_eta = aTrack.momentum().eta();
    float trk_chi2 = aTrack.chi2();
    const auto& stubRefs = aTrack.getStubRefs();
    int nStubs = stubRefs.size();

    float classification = 0.0;  // Default classification is 0

    if (trk_pt >= this->minPt_ && abs(trk_z0) < this->maxZ0_ && abs(trk_eta) < this->maxEta_ &&
        trk_chi2 < this->chi2dofMax_ && trk_bend_chi2 < this->bendchi2Max_ && nStubs >= this->nStubsmin_)
      classification = 1.0;
    // Classification updated to 1 if conditions are met

    aTrack.settrkMVA1(classification);
  }

  if ((this->qualityAlgorithm_ == QualityAlgorithm::NN) || (this->qualityAlgorithm_ == QualityAlgorithm::GBDT)) {
    // Setup ONNX input and output names and arrays
    std::vector<std::string> ortinput_names;
    std::vector<std::string> ortoutput_names;

    cms::Ort::FloatArrays ortinput;
    cms::Ort::FloatArrays ortoutputs;

    std::vector<float> Transformed_features = featureTransform(aTrack, this->featureNames_);
    cms::Ort::ONNXRuntime Runtime(this->ONNXmodel_.fullPath());  //Setup ONNX runtime

    ortinput_names.push_back(this->ONNXInputName_);
    ortoutput_names = Runtime.getOutputNames();

    //ONNX runtime recieves a vector of vectors of floats so push back the input
    // vector of float to create a 1,1,21 ortinput
    ortinput.push_back(Transformed_features);

    // batch_size 1 as only one set of transformed features is being processed
    int batch_size = 1;
    // Run classification
    ortoutputs = Runtime.run(ortinput_names, ortinput, {}, ortoutput_names, batch_size);

    if (this->qualityAlgorithm_ == QualityAlgorithm::NN) {
      aTrack.settrkMVA1(ortoutputs[0][0]);
    }

    else if (this->qualityAlgorithm_ == QualityAlgorithm::GBDT) {
      aTrack.settrkMVA1(ortoutputs[1][1]);
    }
    // Slight differences in the ONNX models of the GBDTs and NNs mean different
    // indices of the ortoutput need to be accessed
  }

  else {
    aTrack.settrkMVA1(-999);
  }
}

void L1TrackQuality::setCutParameters(std::string const& AlgorithmString,
                                      float maxZ0,
                                      float maxEta,
                                      float chi2dofMax,
                                      float bendchi2Max,
                                      float minPt,
                                      int nStubmin) {
  qualityAlgorithm_ = QualityAlgorithm::Cut;
  maxZ0_ = maxZ0;
  maxEta_ = maxEta;
  chi2dofMax_ = chi2dofMax;
  bendchi2Max_ = bendchi2Max;
  minPt_ = minPt;
  nStubsmin_ = nStubmin;
}

void L1TrackQuality::setONNXModel(std::string const& AlgorithmString,
                                  edm::FileInPath const& ONNXmodel,
                                  std::string const& ONNXInputName,
                                  std::vector<std::string> const& featureNames) {
  //Convert algorithm string to Enum class for track by track comparison
  if (AlgorithmString == "NN") {
    qualityAlgorithm_ = QualityAlgorithm::NN;
  } else if (AlgorithmString == "GBDT") {
    qualityAlgorithm_ = QualityAlgorithm::GBDT;
  } else {
    qualityAlgorithm_ = QualityAlgorithm::None;
  }
  ONNXmodel_ = ONNXmodel;
  ONNXInputName_ = ONNXInputName;
  featureNames_ = featureNames;
}
