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
#include <memory>

#include "conifer.h"
#include "ap_fixed.h"

class L1TrackQuality {
public:
  //Default Constructor
  L1TrackQuality();

  L1TrackQuality(const edm::ParameterSet& qualityParams);

  //Default Destructor
  ~L1TrackQuality() = default;

  // Controls the conversion between TTTrack features and ML model training features
  std::vector<float> featureTransform(TTTrack<Ref_Phase2TrackerDigi_>& aTrack,
                                      std::vector<std::string> const& featureNames);

  // Passed by reference a track without MVA filled, method fills the track's MVA field
  double getL1TrackQuality(TTTrack<Ref_Phase2TrackerDigi_>& aTrack);
  // Function to run the BDT in isolation allowing a feature vector in the ap_fixed datatype to be passed
  // and a single output to be returned which is then used to fill the bits in the Track Word for situations
  // where a TTTrack datatype is unavailable to be passed to the track quality
  float runEmulatedTQ(std::vector<ap_fixed<10, 5>> inputFeatures);

  void setModel(edm::FileInPath const& model, std::vector<std::string> const& featureNames);

  void setBonusFeatures(std::vector<float> bonusFeatures);

  // TQ MVA bin conversions
  static constexpr double invSigmoid(double value) { return -log(1. / value - 1.); }
  static constexpr std::array<double, 1 << TTTrack_TrackWord::TrackBitWidths::kMVAQualitySize> getTqMVAPreSigBins() {
    return {{-16.,
             invSigmoid(TTTrack_TrackWord::tqMVABins[1]),
             invSigmoid(TTTrack_TrackWord::tqMVABins[2]),
             invSigmoid(TTTrack_TrackWord::tqMVABins[3]),
             invSigmoid(TTTrack_TrackWord::tqMVABins[4]),
             invSigmoid(TTTrack_TrackWord::tqMVABins[5]),
             invSigmoid(TTTrack_TrackWord::tqMVABins[6]),
             invSigmoid(TTTrack_TrackWord::tqMVABins[7])}};
  }

private:
  // Private Member Data
  edm::FileInPath model_;
  std::vector<std::string> featureNames_;
  bool useHPH_;
  std::vector<float> bonusFeatures_;
};
#endif
