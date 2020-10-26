/*
Track Quality Body file

C.Brown & C.Savard 07/2020
*/

#include "L1Trigger/TrackTrigger/interface/TrackQuality.h"

//Constructors

TrackQuality::TrackQuality() {}

TrackQuality::TrackQuality(edm::ParameterSet& qualityParams) {
  std::string qualityAlgorithm = qualityParams.getParameter<std::string>("qualityAlgorithm");
  // Unpacks EDM parameter set itself to save unecessary processing within TrackProducers
  if (qualityAlgorithm == "Cut") {
    setCutParameters(qualityAlgorithm,
                     (float)qualityParams.getParameter<double>("maxZ0"),
                     (float)qualityParams.getParameter<double>("maxEta"),
                     (float)qualityParams.getParameter<double>("chi2dofMax"),
                     (float)qualityParams.getParameter<double>("bendchi2Max"),
                     (float)qualityParams.getParameter<double>("minPt"),
                     qualityParams.getParameter<int>("nStubsmin"));
  }

  else {
    setONNXModel(qualityAlgorithm,
                 edm::FileInPath(qualityParams.getParameter<std::string>("ONNXmodel")),
                 qualityParams.getParameter<std::string>("ONNXInputName"),
                 qualityParams.getParameter<std::vector<std::string>>("featureNames"));
  }
}

std::vector<float> TrackQuality::featureTransform(TTTrack<Ref_Phase2TrackerDigi_>& aTrack,
                                                  std::vector<std::string> const& featureNames) {
  // List input features for MVA in proper order below, the features options are
  // {"log_chi2","log_chi2rphi","log_chi2rz","log_bendchi2","nstubs","lay1_hits","lay2_hits",
  // "lay3_hits","lay4_hits","lay5_hits","lay6_hits","disk1_hits","disk2_hits","disk3_hits",
  // "disk4_hits","disk5_hits","rinv","tanl","z0","dtot","ltot","chi2","chi2rz","chi2rphi",
  // "bendchi2","pt","eta","nlaymiss_interior"}

  std::vector<float> transformedFeatures;

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

  for (int j = 0; j < eta_size; j++) {
    if (eta >= eta_bins[j] && eta < eta_bins[j + 1])  // if track in eta bin
    {
      // Iterate through hitpattern binary
      for (int k = 0; k <= 6; k++)
        // Fill expanded binary entries using the expected hitmap table positions
        hitpattern_expanded_binary[hitmap[j][k]] = hitpattern_binary[k];
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

  // While not strictly necessary to define these parameters,
  // it is included so each variable is named to avoid confusion
  float tmp_trk_big_invr = 500 * abs(aTrack.rInv());
  float tmp_trk_tanl = abs(aTrack.tanL());
  float tmp_trk_z0 = abs(aTrack.z0());
  float tmp_trk_pt = aTrack.momentum().perp();
  float tmp_trk_eta = aTrack.eta();
  float tmp_trk_chi2 = aTrack.chi2();
  float tmp_trk_chi2rphi = aTrack.chi2XY();
  float tmp_trk_chi2rz = aTrack.chi2Z();
  float tmp_trk_bendchi2 = aTrack.stubPtConsistency();
  float tmp_trk_log_chi2 = log(tmp_trk_chi2);
  float tmp_trk_log_chi2rphi = log(tmp_trk_chi2rphi);
  float tmp_trk_log_chi2rz = log(tmp_trk_chi2rz);
  float tmp_trk_log_bendchi2 = log(tmp_trk_bendchi2);

  // fill feature map
  std::map<std::string, float> feature_map;
  feature_map["log_chi2"] = tmp_trk_log_chi2;
  feature_map["log_chi2rphi"] = tmp_trk_log_chi2rphi;
  feature_map["log_chi2rz"] = tmp_trk_log_chi2rz;
  feature_map["log_bendchi2"] = tmp_trk_log_bendchi2;
  feature_map["chi2"] = tmp_trk_chi2;
  feature_map["chi2rphi"] = tmp_trk_chi2rphi;
  feature_map["chi2rz"] = tmp_trk_chi2rz;
  feature_map["bendchi2"] = tmp_trk_bendchi2;
  feature_map["nstubs"] = float(tmp_trk_dtot + tmp_trk_ltot);
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
  feature_map["rinv"] = tmp_trk_big_invr;
  feature_map["tanl"] = tmp_trk_tanl;
  feature_map["z0"] = tmp_trk_z0;
  feature_map["dtot"] = float(tmp_trk_dtot);
  feature_map["ltot"] = float(tmp_trk_ltot);
  feature_map["pt"] = tmp_trk_pt;
  feature_map["eta"] = tmp_trk_eta;
  feature_map["nlaymiss_interior"] = float(tmp_trk_nlaymiss_interior);

  // fill tensor with track params
  transformedFeatures.reserve(featureNames.size());
  for (const std::string& feature : featureNames)
    transformedFeatures.push_back(feature_map[feature]);

  return transformedFeatures;
}

void TrackQuality::setTrackQuality(TTTrack<Ref_Phase2TrackerDigi_>& aTrack) {
  if (this->qualityAlgorithm_ == "Cut") {
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

  if ((this->qualityAlgorithm_ == "NN") || (this->qualityAlgorithm_ == "GBDT")) {
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

    if (this->qualityAlgorithm_ == "NN") {
      aTrack.settrkMVA1(ortoutputs[0][0]);
    }

    if (this->qualityAlgorithm_ == "GBDT") {
      aTrack.settrkMVA1(ortoutputs[1][1]);
    }
    // Slight differences in the ONNX models of the GBDTs and NNs mean different
    // indices of the ortoutput need to be accessed
  }

  else {
    aTrack.settrkMVA1(-999);
  }
}

void TrackQuality::setCutParameters(std::string const& qualityAlgorithm,
                                    float maxZ0,
                                    float maxEta,
                                    float chi2dofMax,
                                    float bendchi2Max,
                                    float minPt,
                                    int nStubmin) {
  qualityAlgorithm_ = qualityAlgorithm;
  maxZ0_ = maxZ0;
  maxEta_ = maxEta;
  chi2dofMax_ = chi2dofMax;
  bendchi2Max_ = bendchi2Max;
  minPt_ = minPt;
  nStubsmin_ = nStubmin;
}

void TrackQuality::setONNXModel(std::string const& qualityAlgorithm,
                                edm::FileInPath const& ONNXmodel,
                                std::string const& ONNXInputName,
                                std::vector<std::string> const& featureNames) {
  qualityAlgorithm_ = qualityAlgorithm;
  ONNXmodel_ = ONNXmodel;
  ONNXInputName_ = ONNXInputName;
  featureNames_ = featureNames;
}
