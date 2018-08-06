#include "Calibration/IsolatedParticles/interface/CaloConstants.h"
#include "Calibration/IsolatedParticles/interface/ChargeIsolationExtra.h"
#include "Calibration/IsolatedParticles/interface/ChargeIsolation.h"
#include "Calibration/IsolatedParticles/interface/CaloPropagateTrack.h"
#include "Calibration/IsolatedParticles/interface/FindDistCone.h"
#include "Calibration/IsolatedParticles/interface/MatrixECALDetIds.h"
#include "Calibration/IsolatedParticles/interface/MatrixHCALDetIds.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"

#include<iostream>

//#define EDM_ML_DEBUG

namespace spr{

  double chargeIsolation(const edm::Event& iEvent, 
			 const edm::EventSetup& iSetup, 
			 CaloNavigator<DetId>& theNavigator,
			 reco::TrackCollection::const_iterator trkItr,
			 edm::Handle<reco::TrackCollection> trkCollection, 
			 const CaloSubdetectorGeometry* gEB, 
			 const CaloSubdetectorGeometry* gEE, 
			 TrackDetectorAssociator& associator, 
			 TrackAssociatorParameters& parameters_, int ieta, 
			 int iphi, const std::string & theTrackQuality, bool
#ifdef EDM_ML_DEBUG
			 debug
#endif
			 ) {
  
    double maxNearP = -1.0;
    reco::TrackBase::TrackQuality trackQuality_=reco::TrackBase::qualityByName(theTrackQuality);

    // const DetId anyCell,
    reco::TrackCollection::const_iterator trkItr2;
    for (trkItr2 = trkCollection->begin(); trkItr2 != trkCollection->end(); ++trkItr2) {

      const reco::Track* pTrack2 = &(*trkItr2);

      bool   trkQuality  = pTrack2->quality(trackQuality_);
      if ( (trkItr2 != trkItr) && trkQuality )  {
      
	const FreeTrajectoryState fts2 = associator.getFreeTrajectoryState(iSetup, *pTrack2);
	TrackDetMatchInfo info2 = associator.associate(iEvent, iSetup, fts2, parameters_);
	const GlobalPoint point2(info2.trkGlobPosAtEcal.x(),info2.trkGlobPosAtEcal.y(),info2.trkGlobPosAtEcal.z());

	if (info2.isGoodEcal ) {
	  if (std::abs(point2.eta())<spr::etaBEEcal) {
	    const DetId anyCell = gEB->getClosestCell(point2);
#ifdef EDM_ML_DEBUG
	    if (debug) std::cout << "chargeIsolation:: EB cell " << (EBDetId)(anyCell) << " for pt " << pTrack2->p() << std::endl;
#endif
	    if (!spr::chargeIsolation(anyCell,theNavigator,ieta, iphi)) {
	      if (maxNearP<pTrack2->p()) maxNearP=pTrack2->p();
	    }
	  } else {
	    const DetId anyCell = gEE->getClosestCell(point2);
#ifdef EDM_ML_DEBUG
	    if (debug) std::cout << "chargeIsolation:: EE cell " << (EEDetId)(anyCell) << " for pt " << pTrack2->p() << std::endl;
#endif
	    if(!spr::chargeIsolation(anyCell,theNavigator,ieta, iphi)) {
	      if (maxNearP<pTrack2->p()) maxNearP=pTrack2->p();
	    }
	  }
	} //info2.isGoodEcal
      }
    }
    return maxNearP;
  }

  //===========================================================================================================

  bool chargeIsolation(const DetId anyCell,CaloNavigator<DetId>& navigator,int ieta,int iphi) {

    bool isIsolated = false;

    DetId thisDet;

    for (int dx = -ieta; dx < ieta+1; ++dx) {
      for (int dy = -iphi; dy < iphi+1; ++dy) {

	thisDet = navigator.offsetBy(dx, dy);
	navigator.home();
      
	if (thisDet != DetId(0)) {
	  if (thisDet == anyCell) {
	    isIsolated = false;
	    return isIsolated;
	  }
	}
      }
    }
    return isIsolated;
  }

  //===========================================================================================================

  double chargeIsolationEcal(const edm::Event& iEvent, const edm::EventSetup& iSetup, const DetId& coreDet, reco::TrackCollection::const_iterator trkItr, edm::Handle<reco::TrackCollection> trkCollection, const CaloGeometry* geo, const CaloTopology* caloTopology, TrackDetectorAssociator& associator, TrackAssociatorParameters& parameters_, int ieta, int iphi, const std::string& theTrackQuality, bool debug) {
  
    const CaloSubdetectorGeometry *barrelGeom = (geo->getSubdetectorGeometry(DetId::Ecal,EcalBarrel));
    const CaloSubdetectorGeometry *endcapGeom = (geo->getSubdetectorGeometry(DetId::Ecal,EcalEndcap));

    std::vector<DetId> vdets = spr::matrixECALIds(coreDet, ieta, iphi, geo, caloTopology, debug);
#ifdef EDM_ML_DEBUG
    if (debug) std::cout << "chargeIsolation:: eta/phi/dets " << ieta << " " << iphi << " " << vdets.size() << std::endl;
#endif
    double maxNearP = -1.0;
    reco::TrackBase::TrackQuality trackQuality_=reco::TrackBase::qualityByName(theTrackQuality);

    // const DetId anyCell,
    reco::TrackCollection::const_iterator trkItr2;
    for (trkItr2 = trkCollection->begin(); trkItr2 != trkCollection->end(); ++trkItr2) {

      const reco::Track* pTrack2 = &(*trkItr2);

      bool   trkQuality  = pTrack2->quality(trackQuality_);
      if ( (trkItr2 != trkItr) && trkQuality )  {
      
	const FreeTrajectoryState fts2 = associator.getFreeTrajectoryState(iSetup, *pTrack2);
	TrackDetMatchInfo info2 = associator.associate(iEvent, iSetup, fts2, parameters_);
	const GlobalPoint point2(info2.trkGlobPosAtEcal.x(),info2.trkGlobPosAtEcal.y(),info2.trkGlobPosAtEcal.z());

	if (info2.isGoodEcal ) {
	  if (std::abs(point2.eta())<spr::etaBEEcal) {
	    const DetId anyCell = barrelGeom->getClosestCell(point2);
#ifdef EDM_ML_DEBUG
	    if (debug) std::cout << "chargeIsolation:: EB cell " << (EBDetId)(anyCell) << " for pt " << pTrack2->p() << std::endl;
#endif
	    if (!spr::chargeIsolation(anyCell,vdets)) {
	      if (maxNearP<pTrack2->p()) maxNearP=pTrack2->p();
	    }
	  } else {
	    const DetId anyCell = endcapGeom->getClosestCell(point2);
#ifdef EDM_ML_DEBUG
	    if (debug) std::cout << "chargeIsolation:: EE cell " << (EEDetId)(anyCell) << " for pt " << pTrack2->p() << std::endl;
#endif
	    if (!spr::chargeIsolation(anyCell,vdets)) {
	      if (maxNearP<pTrack2->p()) maxNearP=pTrack2->p();
	    }
	  }
	} //info2.isGoodEcal
      }
    }
    return maxNearP;
  }

  //===========================================================================================================

  double chargeIsolationHcal(const edm::Event& iEvent, const edm::EventSetup& iSetup, reco::TrackCollection::const_iterator trkItr, edm::Handle<reco::TrackCollection> trkCollection, const DetId ClosestCell, const HcalTopology* topology, const CaloSubdetectorGeometry* gHB, TrackDetectorAssociator& associator, TrackAssociatorParameters& parameters_, int ieta, int iphi, const std::string& theTrackQuality, bool debug) {

    std::vector<DetId> dets(1,ClosestCell);

#ifdef EDM_ML_DEBUG
    if (debug) std::cout << (HcalDetId) ClosestCell << std::endl;
#endif
    std::vector<DetId> vdets = spr::matrixHCALIds(dets, topology, ieta, iphi, false, debug);
  
#ifdef EDM_ML_DEBUG
    if (debug) {
      for (unsigned int i=0; i<vdets.size(); i++) {
	std::cout << "HcalDetId in " <<2*ieta+1 << "x" << 2*iphi+1 << " " << (HcalDetId) vdets[i] << std::endl;
      }
    }
#endif
    double maxNearP = -1.0;
    reco::TrackBase::TrackQuality trackQuality_=reco::TrackBase::qualityByName(theTrackQuality);
  
    reco::TrackCollection::const_iterator trkItr2;
    for (trkItr2 = trkCollection->begin(); trkItr2 != trkCollection->end(); ++trkItr2) {
    
      const reco::Track* pTrack2 = &(*trkItr2);
    
      bool   trkQuality  = pTrack2->quality(trackQuality_);
      if ( (trkItr2 != trkItr) && trkQuality )  {
	const FreeTrajectoryState fts2 = associator.getFreeTrajectoryState(iSetup, *pTrack2);
	TrackDetMatchInfo info2 = associator.associate(iEvent, iSetup, fts2, parameters_);
	const GlobalPoint point2(info2.trkGlobPosAtHcal.x(),info2.trkGlobPosAtHcal.y(),info2.trkGlobPosAtHcal.z());

#ifdef EDM_ML_DEBUG
	if (debug) {
	  std::cout << "Track2 (p,eta,phi) " << pTrack2->p() << " " << pTrack2->eta() << " " << pTrack2->phi() << std::endl;
	}
#endif
	if (info2.isGoodHcal ) {
	  const DetId anyCell = gHB->getClosestCell(point2);
#ifdef EDM_ML_DEBUG
	  if (debug) std::cout << "chargeIsolation:: HCAL cell " << (HcalDetId)(anyCell) << " for pt " << pTrack2->p() << std::endl;
#endif
	  if (!spr::chargeIsolation(anyCell,vdets)) {	
	    if(maxNearP<pTrack2->p())  maxNearP=pTrack2->p();
	  }
#ifdef EDM_ML_DEBUG
	  if (debug){
	    std::cout << "maxNearP " << maxNearP << " thisCell " 
		      << (HcalDetId)anyCell << " (" 
		      << info2.trkGlobPosAtHcal.x() << ","
		      << info2.trkGlobPosAtHcal.y() <<","
		      << info2.trkGlobPosAtHcal.z() <<")" << std::endl;
	  }
#endif
	}
      }
    }
    return maxNearP;
  }

}
