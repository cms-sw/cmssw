#ifndef RecoMuon_TrackerSeedGenerator_L1MuonRegionProducer_H
#define RecoMuon_TrackerSeedGenerator_L1MuonRegionProducer_H

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include <vector>

class TrackingRegion;
class L1MuGMTCand;
namespace edm { class Event; class EventSetup; class ParameterSet; }

class L1MuonRegionProducer : public TrackingRegionProducer {

public:
  L1MuonRegionProducer(const edm::ParameterSet& cfg,
	   edm::ConsumesCollector && iC);
  virtual ~L1MuonRegionProducer(){} 
  void setL1Constraint(const L1MuGMTCand & muon); 
  virtual std::vector<std::unique_ptr<TrackingRegion> >
      regions(const edm::Event& ev, const edm::EventSetup& es) const override;

private:
  // region configuration
  double thePtMin, theOriginRadius, theOriginHalfLength; 
  GlobalPoint theOrigin;

  // L1 constraint
  double thePtL1, thePhiL1, theEtaL1; int theChargeL1;

};

#endif
