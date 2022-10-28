#ifndef RecoMuon_TrackerSeedGenerator_L1MuonRegionProducer_H
#define RecoMuon_TrackerSeedGenerator_L1MuonRegionProducer_H
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/FrameworkfwdMostUsed.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include <vector>
#include <memory>

class TrackingRegion;
class L1MuGMTCand;
class MagneticField;
class IdealMagneticFieldRecord;
class MultipleScatteringParametrisationMaker;
class TrackerMultipleScatteringRecord;

class L1MuonRegionProducer {
public:
  L1MuonRegionProducer(const edm::ParameterSet& cfg, edm::ConsumesCollector iC);
  ~L1MuonRegionProducer() = default;
  void setL1Constraint(const L1MuGMTCand& muon);
  std::vector<std::unique_ptr<TrackingRegion> > regions(const edm::EventSetup& iSetup) const;

private:
  // region configuration
  double thePtMin, theOriginRadius, theOriginHalfLength;
  GlobalPoint theOrigin;

  // L1 constraint
  double thePtL1, thePhiL1, theEtaL1;
  int theChargeL1;

  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> theFieldToken;
  edm::ESGetToken<MultipleScatteringParametrisationMaker, TrackerMultipleScatteringRecord> theMSMakerToken;
};

#endif
