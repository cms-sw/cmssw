/*
Track Quality Body file
C.Brown & C.Savard 07/2020
*/

#include "L1Trigger/TrackTrigger/interface/L1TrackQuality.h"

//Constructors

L1TrackQuality::L1TrackQuality() {}

L1TrackQuality::L1TrackQuality(const edm::ParameterSet& qualityParams) : useHPH_(false), bonusFeatures_() {
  edm::FileInPath modelPath = qualityParams.getParameter<edm::FileInPath>("model");
  std::string fullPath = modelPath.fullPath();

  if (fullPath.substr(fullPath.length() - 4) == "onnx") {
    setModel(qualityParams.getParameter<edm::FileInPath>("model"),
             qualityParams.getParameter<std::vector<std::string>>("featureNames"),
             qualityParams.getParameter<std::string>("ONNXInputName"));
    runTime_ = std::make_unique<cms::Ort::ONNXRuntime>(this->model_.fullPath());
  } else {
    // Unpacks EDM parameter set itself to save unecessary processing within TrackProducers
    setModel(qualityParams.getParameter<edm::FileInPath>("model"),
             qualityParams.getParameter<std::vector<std::string>>("featureNames"));
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
  float tmp_trk_z0_scaled = tmp_trk_z0 / abs(aTrack.minZ0);
  float tmp_trk_phi = aTrack.phi();
  float tmp_trk_eta = aTrack.eta();
  float tmp_trk_tanl = aTrack.tanL();
  float tmp_trk_d0 = aTrack.d0();

  // -------- fill the feature map ---------

  feature_map["nstub"] = float(tmp_trk_nstub);
  feature_map["z0"] = tmp_trk_z0;
  feature_map["z0_scaled"] = tmp_trk_z0_scaled;
  feature_map["phi"] = tmp_trk_phi;
  feature_map["eta"] = tmp_trk_eta;
  feature_map["nlaymiss_interior"] = float(tmp_trk_nlaymiss_interior);
  feature_map["bendchi2_bin"] = tmp_trk_bendchi2_bin;
  feature_map["chi2rphi_bin"] = tmp_trk_chi2rphi_bin;
  feature_map["chi2rz_bin"] = tmp_trk_chi2rz_bin;
  feature_map["tanl"] = tmp_trk_tanl;
  feature_map["d0"] = tmp_trk_d0;

  // fill tensor with track params
  transformedFeatures.reserve(featureNames.size());
  for (const std::string& feature : featureNames)
    transformedFeatures.push_back(feature_map[feature]);

  return transformedFeatures;
}

double L1TrackQuality::getL1TrackQuality(TTTrack<Ref_Phase2TrackerDigi_>& aTrack) {
  std::string fullPath = this->model_.fullPath();
  if (fullPath.substr(fullPath.length() - 4) == "onnx") {
    // Setup ONNX input and output names and arrays
    std::vector<std::string> ortinput_names;
    std::vector<std::string> ortoutput_names;

    cms::Ort::FloatArrays ortinput;
    cms::Ort::FloatArrays ortoutputs;

    std::vector<float> Transformed_features = featureTransform(aTrack, this->featureNames_);

    ortinput_names.push_back(this->inputName_);
    ortoutput_names = runTime_->getOutputNames();

    //ONNX runtime recieves a vector of vectors of floats so push back the input
    // vector of float to create a 1,1,21 ortinput
    ortinput.push_back(Transformed_features);

    // batch_size 1 as only one set of transformed features is being processed
    int batch_size = 1;
    // Run classification
    ortoutputs = runTime_->run(ortinput_names, ortinput, {}, ortoutput_names, batch_size);
    return ortoutputs[1][1];
  }

  // load in bdt
  conifer::BDT<float, float> bdt(this->model_.fullPath());

  // collect features and classify using bdt
  std::vector<float> inputs = featureTransform(aTrack, this->featureNames_);
  std::vector<float> output = bdt.decision_function(inputs);
  return (1. / (1. + exp(-output.at(0))));
}

float L1TrackQuality::runEmulatedTQ(std::vector<ap_fixed<10, 5>> inputFeatures) {
  // load in bdt

  conifer::BDT<ap_fixed<10, 5>, ap_fixed<10, 5>> bdt(this->model_.fullPath());

  // collect features and classify using bdt
  std::vector<ap_fixed<10, 5>> output = bdt.decision_function(inputFeatures);
  return output.at(0).to_float();  // need logistic sigmoid fcn applied to xgb output
}

void L1TrackQuality::setModel(edm::FileInPath const& model,
                              std::vector<std::string> const& featureNames,
                              std::string const& inputName) {
  //Convert algorithm string to Enum class for track by track comparison
  model_ = model;
  featureNames_ = featureNames;
  inputName_ = inputName;
}

void L1TrackQuality::setBonusFeatures(std::vector<float> bonusFeatures) {
  bonusFeatures_ = bonusFeatures;
  useHPH_ = true;
}
