#include "RecoMuon/TrackerSeedGenerator/interface/L1MuonRegionProducer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"
#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"
#include "RecoMuon/TrackerSeedGenerator/interface/L1MuonPixelTrackFitter.h"

using namespace std;

L1MuonRegionProducer::L1MuonRegionProducer(const edm::ParameterSet& cfg,
	   edm::ConsumesCollector && iC) { 

  edm::ParameterSet regionPSet = cfg.getParameter<edm::ParameterSet>("RegionPSet");

  thePtMin            = regionPSet.getParameter<double>("ptMin");
  theOriginRadius     = regionPSet.getParameter<double>("originRadius");
  theOriginHalfLength = regionPSet.getParameter<double>("originHalfLength");
  theOrigin = GlobalPoint( regionPSet.getParameter<double>("originXPos"),
                           regionPSet.getParameter<double>("originYPos"),
                           regionPSet.getParameter<double>("originZPos"));
}   

void L1MuonRegionProducer::setL1Constraint(const L1MuGMTCand & muon)
{
  thePhiL1 = muon.phiValue()+0.021817;
  theEtaL1 = muon.etaValue();
  theChargeL1 = muon.charge();
}

std::vector<std::unique_ptr<TrackingRegion> > L1MuonRegionProducer::
      regions(const edm::Event& ev, const edm::EventSetup& es) const
{
  double dx = cos(thePhiL1);
  double dy = sin(thePhiL1);
  double dz = sinh(theEtaL1);
  GlobalVector direction(dx,dy,dz);        // muon direction

  std::vector<std::unique_ptr<TrackingRegion> > result;
  double bending = L1MuonPixelTrackFitter::getBending(1./thePtMin, theEtaL1, theChargeL1);
  bending = fabs(bending);
  double errBending = L1MuonPixelTrackFitter::getBendingError(1./thePtMin, theEtaL1);

  result.push_back( 
      std::make_unique<RectangularEtaPhiTrackingRegion>( direction, theOrigin,
          thePtMin, theOriginRadius, theOriginHalfLength, 0.15, bending+3*errBending) );

  return result;
}




