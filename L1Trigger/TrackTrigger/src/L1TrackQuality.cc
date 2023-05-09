/*
Track Quality Body file
C.Brown & C.Savard 07/2020
*/

#include "L1Trigger/TrackTrigger/interface/L1TrackQuality.h"

//Constructors

L1TrackQuality::L1TrackQuality() {}

L1TrackQuality::L1TrackQuality(const edm::ParameterSet& qualityParams) : useHPH_(false), bonusFeatures_() {
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
    if ((AlgorithmString == "GBDT") || (AlgorithmString == "NN"))
      runTime_ = std::make_unique<cms::Ort::ONNXRuntime>(this->ONNXmodel_.fullPath());
  }
}

std::vector<float> L1TrackQuality::featureTransform(TTTrack<Ref_Phase2TrackerDigi_>& aTrack,
                                                    std::vector<std::string> const& featureNames) {
  // List input features for MVA in proper order below, the current features options are
  // {"phi", "eta", "z0", "bendchi2_bin", "nstub", "nlaymiss_interior", "chi2rphi_bin",
  // "chi2rz_bin"}
  //
  // To use more features, they must be created here and added to feature_map below

  std::vector<float> transformedFeatures;

  // Define feature map, filled as features are generated
  std::map<std::string, float> feature_map;

  // -------- calculate feature variables --------

  // calculate number of missed interior layers from hitpattern
  int tmp_trk_hitpattern = aTrack.hitPattern();
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

  // binned chi2 variables
  int tmp_trk_bendchi2_bin = aTrack.getBendChi2Bits();
  int tmp_trk_chi2rphi_bin = aTrack.getChi2RPhiBits();
  int tmp_trk_chi2rz_bin = aTrack.getChi2RZBits();

  // get the nstub
  std::vector<edm::Ref<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>, TTStub<Ref_Phase2TrackerDigi_>>> stubRefs =
      aTrack.getStubRefs();
  int tmp_trk_nstub = stubRefs.size();

  // get other variables directly from TTTrack
  float tmp_trk_z0 = aTrack.z0();
  float tmp_trk_phi = aTrack.phi();
  float tmp_trk_eta = aTrack.eta();

  // -------- fill the feature map ---------

  feature_map["nstub"] = float(tmp_trk_nstub);
  feature_map["z0"] = tmp_trk_z0;
  feature_map["phi"] = tmp_trk_phi;
  feature_map["eta"] = tmp_trk_eta;
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

  else if (this->qualityAlgorithm_ == QualityAlgorithm::GBDT_cpp) {
    // load in bdt
    conifer::BDT<float, float> bdt(this->ONNXmodel_.fullPath());

    // collect features and classify using bdt
    std::vector<float> inputs = featureTransform(aTrack, this->featureNames_);
    std::vector<float> output = bdt.decision_function(inputs);
    aTrack.settrkMVA1(1. / (1. + exp(-output.at(0))));  // need logistic sigmoid fcn applied to xgb output
  }

  else if ((this->qualityAlgorithm_ == QualityAlgorithm::NN) || (this->qualityAlgorithm_ == QualityAlgorithm::GBDT)) {
    // Setup ONNX input and output names and arrays
    std::vector<std::string> ortinput_names;
    std::vector<std::string> ortoutput_names;

    cms::Ort::FloatArrays ortinput;
    cms::Ort::FloatArrays ortoutputs;

    std::vector<float> Transformed_features = featureTransform(aTrack, this->featureNames_);
    //    cms::Ort::ONNXRuntime runTime(this->ONNXmodel_.fullPath());  //Setup ONNX runtime

    ortinput_names.push_back(this->ONNXInputName_);
    ortoutput_names = runTime_->getOutputNames();

    //ONNX runtime recieves a vector of vectors of floats so push back the input
    // vector of float to create a 1,1,21 ortinput
    ortinput.push_back(Transformed_features);

    // batch_size 1 as only one set of transformed features is being processed
    int batch_size = 1;
    // Run classification
    ortoutputs = runTime_->run(ortinput_names, ortinput, {}, ortoutput_names, batch_size);

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

float L1TrackQuality::runEmulatedTQ(std::vector<ap_fixed<10, 5>> inputFeatures) {
  // load in bdt

  conifer::BDT<ap_fixed<10, 5>, ap_fixed<10, 5>> bdt(this->ONNXmodel_.fullPath());

  // collect features and classify using bdt
  std::vector<ap_fixed<10, 5>> output = bdt.decision_function(inputFeatures);
  return output.at(0).to_float();  // need logistic sigmoid fcn applied to xgb output
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
  } else if (AlgorithmString == "GBDT_cpp") {
    qualityAlgorithm_ = QualityAlgorithm::GBDT_cpp;
  } else {
    qualityAlgorithm_ = QualityAlgorithm::None;
  }
  ONNXmodel_ = ONNXmodel;
  ONNXInputName_ = ONNXInputName;
  featureNames_ = featureNames;
}

void L1TrackQuality::setBonusFeatures(std::vector<float> bonusFeatures) {
  bonusFeatures_ = bonusFeatures;
  useHPH_ = true;
}
