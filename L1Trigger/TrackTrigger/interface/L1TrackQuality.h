/*
Track Quality Header file
C.Brown 28/07/20
*/

#ifndef L1Trigger_TrackTrigger_interface_L1TrackQuality_h
#define L1Trigger_TrackTrigger_interface_L1TrackQuality_h

#include <iostream>
#include <set>
#include <vector>
#include <memory>
#include <string>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack_TrackWord.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"
#include <memory>

#include "conifer.h"
#include "ap_fixed.h"

class L1TrackQuality {
public:
  // Enum class used for determining prediction behaviour in setL1TrackQuality
  enum class QualityAlgorithm { Cut, GBDT, GBDT_cpp, NN, None };

  //Default Constructor
  L1TrackQuality();

  L1TrackQuality(const edm::ParameterSet& qualityParams);

  //Default Destructor
  ~L1TrackQuality() = default;

  // Controls the conversion between TTTrack features and ML model training features
  std::vector<float> featureTransform(TTTrack<Ref_Phase2TrackerDigi_>& aTrack,
                                      std::vector<std::string> const& featureNames);

  // Passed by reference a track without MVA filled, method fills the track's MVA field
  void setL1TrackQuality(TTTrack<Ref_Phase2TrackerDigi_>& aTrack);

  float runEmulatedTQ(std::vector<ap_fixed<10, 5>> inputFeatures);
  // To set private member data
  void setCutParameters(std::string const& AlgorithmString,
                        float maxZ0,
                        float maxEta,
                        float chi2dofMax,
                        float bendchi2Max,
                        float minPt,
                        int nStubmin);

  void setONNXModel(std::string const& AlgorithmString,
                    edm::FileInPath const& ONNXmodel,
                    std::string const& ONNXInputName,
                    std::vector<std::string> const& featureNames);

  void setBonusFeatures(std::vector<float> bonusFeatures);

private:
  // Private Member Data
  QualityAlgorithm qualityAlgorithm_ = QualityAlgorithm::None;
  edm::FileInPath ONNXmodel_;
  std::string ONNXInputName_;
  std::vector<std::string> featureNames_;
  float maxZ0_;
  float maxEta_;
  float chi2dofMax_;
  float bendchi2Max_;
  float minPt_;
  int nStubsmin_;
  bool useHPH_;
  std::vector<float> bonusFeatures_;
  std::unique_ptr<cms::Ort::ONNXRuntime> runTime_;
};
#endif
