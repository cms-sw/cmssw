#ifndef HLTmmkFilter_h
#define HLTmmkFilter_h
//
// Package:    HLTstaging
// Class:      HLTmmkFilter
//
/**\class HLTmmkFilter

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
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
// ----------------------------------------------------------------------

namespace reco {
  class Candidate;
  class Track;
}

class FreeTrajectoryState;
class MagneticField;
	
class HLTmmkFilter : public HLTFilter {

 public:
  explicit HLTmmkFilter(const edm::ParameterSet&);
  ~HLTmmkFilter() override;
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

 private:
  void beginJob() override ;
  bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;
  void endJob() override;

  static int overlap(const reco::Candidate&, const reco::Candidate&);
  static FreeTrajectoryState initialFreeState( const reco::Track&,const MagneticField*);

  edm::InputTag                                          muCandTag_;
  edm::EDGetTokenT<reco::RecoChargedCandidateCollection> muCandToken_;
  edm::InputTag                                          trkCandTag_;
  edm::EDGetTokenT<reco::RecoChargedCandidateCollection> trkCandToken_;

  const double thirdTrackMass_;
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
