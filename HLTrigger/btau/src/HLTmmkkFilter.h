#ifndef HLTmmkkFilter_h
#define HLTmmkkFilter_h
//
// Package:    HLTstaging
// Class:      HLTmmkkFilter
//
/**\class HLTmmkkFilter

 HLT Filter for b to (mumu) + X

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Nicolo Magini
//         Created:  Thu Nov  9 17:55:31 CET 2006
// Modified by Lotte Wilke
// Last Modification: 13.02.2007
//


// system include files
#include <memory>

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
namespace edm {
  class ConfigurationDescriptions;
}
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"

// ----------------------------------------------------------------------

namespace reco {
  class Candidate;
  class Track;
}

class FreeTrajectoryState;
class MagneticField;
	
class HLTmmkkFilter : public HLTFilter {

 public:
  explicit HLTmmkkFilter(const edm::ParameterSet&);
  ~HLTmmkkFilter();
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
 private:
  virtual void beginJob() override;
  virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;
  virtual void endJob() override;

  static int overlap(const reco::Candidate&, const reco::Candidate&);
  static FreeTrajectoryState initialFreeState(const reco::Track &, const MagneticField *);

  edm::InputTag                                          muCandTag_;
  edm::EDGetTokenT<reco::RecoChargedCandidateCollection> muCandToken_;
  edm::InputTag                                          trkCandTag_;
  edm::EDGetTokenT<reco::RecoChargedCandidateCollection> trkCandToken_;

  const double thirdTrackMass_;
  const double fourthTrackMass_;
  const double maxEta_;
  const double minPt_;
  const double minInvMass_;
  const double maxInvMass_;
  const double maxNormalisedChi2_;
  const double minLxySignificance_;
  const double minCosinePointingAngle_;
  const double minD0Significance_;
  const bool fastAccept_;
  edm::InputTag                    beamSpotTag_;
  edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;

};
#endif
