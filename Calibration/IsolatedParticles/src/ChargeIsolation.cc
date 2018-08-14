#include "Calibration/IsolatedParticles/interface/CaloConstants.h"
#include "Calibration/IsolatedParticles/interface/ChargeIsolation.h"
#include "Calibration/IsolatedParticles/interface/CaloPropagateTrack.h"
#include "Calibration/IsolatedParticles/interface/FindDistCone.h"
#include "Calibration/IsolatedParticles/interface/MatrixECALDetIds.h"
#include "Calibration/IsolatedParticles/interface/MatrixHCALDetIds.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"

#include<iostream>

//#define EDM_ML_DEBUG

namespace spr{

  double chargeIsolationEcal(unsigned int trkIndex, std::vector<spr::propagatedTrackID>& vdetIds, const CaloGeometry* geo, const CaloTopology* caloTopology, int ieta, int iphi, bool debug) {
  
    const DetId coreDet = vdetIds[trkIndex].detIdECAL;
#ifdef EDM_ML_DEBUG
    if (debug) {
      if (coreDet.subdetId() == EcalBarrel) 
	std::cout << "DetId " << (EBDetId)(coreDet) << " Flag " << vdetIds[trkIndex].okECAL << std::endl;
      else
	std::cout << "DetId " << (EEDetId)(coreDet) << " Flag " << vdetIds[trkIndex].okECAL << std::endl;
    }
#endif
    double maxNearP = -1.0;
    if (vdetIds[trkIndex].okECAL) {
      std::vector<DetId> vdets = spr::matrixECALIds(coreDet, ieta, iphi, geo, caloTopology, debug);
#ifdef EDM_ML_DEBUG
      if (debug) std::cout << "chargeIsolationEcal:: eta/phi/dets " << ieta << " " << iphi << " " << vdets.size() << std::endl;
#endif

      for (unsigned int indx=0; indx<vdetIds.size(); ++indx) {
	if (indx != trkIndex && vdetIds[indx].ok && vdetIds[indx].okECAL) {
	  const DetId anyCell = vdetIds[indx].detIdECAL;
	  if (!spr::chargeIsolation(anyCell,vdets)) {
	    const reco::Track* pTrack = &(*(vdetIds[indx].trkItr));
	    if (maxNearP < pTrack->p()) maxNearP = pTrack->p();
	  }
	}
      }
    }
    return maxNearP;
  }

  double chargeIsolationEcal(const DetId& coreDet, reco::TrackCollection::const_iterator trkItr, edm::Handle<reco::TrackCollection> trkCollection, const CaloGeometry* geo, const CaloTopology* caloTopology, const MagneticField* bField, int ieta, int iphi, const std::string& theTrackQuality, bool debug) {
  
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

      bool   trkQuality  =  (trackQuality_ != reco::TrackBase::undefQuality) ?
	(pTrack2->quality(trackQuality_)) : true;
      if ( (trkItr2 != trkItr) && trkQuality )  {
      
	std::pair<math::XYZPoint,bool> info = spr::propagateECAL(pTrack2,bField);
	const GlobalPoint point2(info.first.x(),info.first.y(),info.first.z());

	if (info.second) {
	  if (std::abs(point2.eta())<spr::etaBEEcal) {
	    const DetId anyCell = barrelGeom->getClosestCell(point2);
	    if (!spr::chargeIsolation(anyCell,vdets)) {
#ifdef EDM_ML_DEBUG
	      if (debug) std::cout << "chargeIsolationEcal Cell " << (EBDetId)(anyCell) << " pt " << pTrack2->p() << std::endl;
#endif
	      if (maxNearP<pTrack2->p()) maxNearP=pTrack2->p();
	    }
	  } else {
	    if (endcapGeom) {
	      const DetId anyCell = endcapGeom->getClosestCell(point2);
	      if (!spr::chargeIsolation(anyCell,vdets)) {
#ifdef EDM_ML_DEBUG
		if (debug) std::cout << "chargeIsolationEcal Cell " << (EEDetId)(anyCell) << " pt " << pTrack2->p() << std::endl;
#endif
		if (maxNearP<pTrack2->p()) maxNearP=pTrack2->p();
	      }
	    }
	  }
	} //info.second
      }
    }
    return maxNearP;
  }

  double chargeIsolationHcal(unsigned int trkIndex, std::vector<spr::propagatedTrackID> & vdetIds, const HcalTopology* topology, int ieta, int iphi, bool debug) {
  
    std::vector<DetId> dets(1,vdetIds[trkIndex].detIdHCAL);
#ifdef EDM_ML_DEBUG
    if (debug) {
      std::cout << "DetId " << (HcalDetId)(dets[0]) << " Flag " << vdetIds[trkIndex].okHCAL << std::endl;
    }
#endif
    double maxNearP = -1.0;
    if (vdetIds[trkIndex].okHCAL) {
      std::vector<DetId> vdets = spr::matrixHCALIds(dets, topology, ieta, iphi, false, debug);
#ifdef EDM_ML_DEBUG
      if (debug) std::cout << "chargeIsolationHcal:: eta/phi/dets " << ieta << " " << iphi << " " << vdets.size() << std::endl;
#endif
      for (unsigned indx = 0; indx<vdetIds.size(); ++indx) {
	if (indx != trkIndex && vdetIds[indx].ok && vdetIds[indx].okHCAL) {
	  const DetId anyCell = vdetIds[indx].detIdHCAL;
	  if (!spr::chargeIsolation(anyCell,vdets)) {
	    const reco::Track* pTrack = &(*(vdetIds[indx].trkItr));
#ifdef EDM_ML_DEBUG
	    if (debug) std::cout << "chargeIsolationHcal Cell " << (HcalDetId)(anyCell) << " pt " << pTrack->p() << std::endl;
#endif
	    if (maxNearP < pTrack->p()) maxNearP = pTrack->p();
	  }
	}
      }
    }
    return maxNearP;
  }

  //===========================================================================================================

  double chargeIsolationHcal(reco::TrackCollection::const_iterator trkItr, edm::Handle<reco::TrackCollection> trkCollection, const DetId ClosestCell, const HcalTopology* topology, const CaloSubdetectorGeometry* gHB, const MagneticField* bField, int ieta, int iphi, const std::string& theTrackQuality, bool debug) {

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
    
      bool   trkQuality  = (trackQuality_ != reco::TrackBase::undefQuality) ?
	(pTrack2->quality(trackQuality_)) : true;
      if ( (trkItr2 != trkItr) && trkQuality )  {
	std::pair<math::XYZPoint,bool> info = spr::propagateHCAL(pTrack2,bField);
	const GlobalPoint point2(info.first.x(),info.first.y(),info.first.z());

#ifdef EDM_ML_DEBUG
	if (debug) {
	  std::cout << "Track2 (p,eta,phi) " << pTrack2->p() << " " << pTrack2->eta() << " " << pTrack2->phi() << std::endl;
	}
#endif
	if (info.second) {
	  const DetId anyCell = gHB->getClosestCell(point2);
	  if (!spr::chargeIsolation(anyCell,vdets)) {	
	    if(maxNearP<pTrack2->p())  maxNearP=pTrack2->p();
	  }
#ifdef EDM_ML_DEBUG
	  if (debug){
	    std::cout << "maxNearP " << maxNearP << " thisCell " 
		      << (HcalDetId)anyCell << " (" 
		      << info.first.x() << "," << info.first.y() <<","
		      << info.first.z() << ")" << std::endl;
	  }
#endif
	}
      }
    }
    return maxNearP;
  }

  bool chargeIsolation(const DetId anyCell, std::vector<DetId>& vdets) {
    bool isIsolated = true;
    for (unsigned int i=0; i<vdets.size(); i++){
      if (anyCell == vdets[i] ) {
	isIsolated = false;
	break;
      }
    }
    return isIsolated;
  }

  double coneChargeIsolation(const edm::Event& iEvent, const edm::EventSetup& iSetup, reco::TrackCollection::const_iterator trkItr, edm::Handle<reco::TrackCollection> trkCollection, TrackDetectorAssociator& associator, TrackAssociatorParameters& parameters_, const std::string & theTrackQuality, int &nNearTRKs, int &nLayers_maxNearP, int &trkQual_maxNearP, double &maxNearP_goodTrk, const GlobalPoint& hpoint1, const GlobalVector& trackMom, double dR) {

    nNearTRKs=0;
    nLayers_maxNearP=0;
    trkQual_maxNearP=-1; 
    maxNearP_goodTrk = -999.0;
    double maxNearP = -999.0;
    reco::TrackBase::TrackQuality trackQuality_=  reco::TrackBase::qualityByName(theTrackQuality);

    // Iterate over tracks
    reco::TrackCollection::const_iterator trkItr2;
    for( trkItr2 = trkCollection->begin(); 
	 trkItr2 != trkCollection->end(); ++trkItr2){

      // Get track
      const reco::Track* pTrack2 = &(*trkItr2);
    
      // Get track qual, nlayers, and hit pattern
      bool   trkQuality  = (trackQuality_ != reco::TrackBase::undefQuality) ?
	(pTrack2->quality(trackQuality_)) : true;
      if (trkQuality) trkQual_maxNearP  = 1;
      const reco::HitPattern& hitp = pTrack2->hitPattern();
      nLayers_maxNearP = hitp.trackerLayersWithMeasurement() ;        
    
      // Skip if the neighboring track candidate is the iso-track
      // candidate
      if (trkItr2 != trkItr) {
    
	// Get propagator
	const FreeTrajectoryState fts2 = associator.getFreeTrajectoryState(iSetup, *pTrack2);
	TrackDetMatchInfo info2 = associator.associate(iEvent, iSetup, fts2, parameters_);
    
	// Make sure it reaches Hcal
	if (info2.isGoodHcal ) {
    
	  const GlobalPoint point2(info2.trkGlobPosAtHcal.x(),
				   info2.trkGlobPosAtHcal.y(),
				   info2.trkGlobPosAtHcal.z());
    
	  int isConeChargedIso = spr::coneChargeIsolation(hpoint1, point2, trackMom, dR);
    
	  if (isConeChargedIso==0) {
	    nNearTRKs++;
	    if(maxNearP<pTrack2->p()) {
	      maxNearP=pTrack2->p();
	      if (trkQual_maxNearP>0 && nLayers_maxNearP>7 && maxNearP_goodTrk<pTrack2->p()) {
		maxNearP_goodTrk=pTrack2->p();
	      }
	    }
	  }
	}
      }
    } // Iterate over track loop
    
    return maxNearP;
  }

  double chargeIsolationCone(unsigned int trkIndex, std::vector<spr::propagatedTrackDirection> & trkDirs, double dR, int & nNearTRKs, bool debug) {

    double maxNearP = -1.0;
    nNearTRKs = 0;
    if (trkDirs[trkIndex].okHCAL) {
#ifdef EDM_ML_DEBUG
      if (debug) std::cout << "chargeIsolationCone with " << trkDirs.size() << " tracks " << std::endl;
#endif
      for (unsigned int indx=0; indx<trkDirs.size(); ++indx) {
	if (indx != trkIndex && trkDirs[indx].ok && trkDirs[indx].okHCAL) {
	  int isConeChargedIso = spr::coneChargeIsolation(trkDirs[trkIndex].pointHCAL, trkDirs[indx].pointHCAL, trkDirs[trkIndex].directionHCAL, dR);
	  if (isConeChargedIso==0) {
	    nNearTRKs++;
	    const reco::Track* pTrack = &(*(trkDirs[indx].trkItr));
	    if (maxNearP < pTrack->p()) maxNearP = pTrack->p();
	  }
	}
      }
    }
#ifdef EDM_ML_DEBUG
    if (debug) std::cout << "chargeIsolationCone Track " << trkDirs[trkIndex].okHCAL << " maxNearP " << maxNearP << std::endl;
#endif
    return maxNearP;
  }

  std::pair<double,double> chargeIsolationCone(unsigned int trkIndex, std::vector<spr::propagatedTrackDirection> & trkDirs, double dR, bool debug) {

    double maxNearP = -1.0;
    double sumP = 0;
    if (trkDirs[trkIndex].okHCAL) {
#ifdef EDM_ML_DEBUG
      if (debug) std::cout << "chargeIsolationCone with " << trkDirs.size() << " tracks " << std::endl;
#endif
      for (unsigned int indx=0; indx<trkDirs.size(); ++indx) {
	if (indx != trkIndex && trkDirs[indx].ok && trkDirs[indx].okHCAL) {
	  int isConeChargedIso = spr::coneChargeIsolation(trkDirs[trkIndex].pointHCAL, trkDirs[indx].pointHCAL, trkDirs[trkIndex].directionHCAL, dR);
	  if (isConeChargedIso==0) {
	    const reco::Track* pTrack = &(*(trkDirs[indx].trkItr));
	    sumP += (pTrack->p());
	    if (maxNearP < pTrack->p()) maxNearP = pTrack->p();
	  }
	}
      }
    }
#ifdef EDM_ML_DEBUG
    if (debug) std::cout << "chargeIsolationCone Track " << trkDirs[trkIndex].okHCAL << " maxNearP " << maxNearP << ":" << sumP <<std::endl;
#endif
    return std::pair<double,double>(maxNearP,sumP);
  }

  int coneChargeIsolation(const GlobalPoint& hpoint1, const GlobalPoint& point2, const GlobalVector& trackMom, double dR) {			 

    int isIsolated = 1;
    if (spr::getDistInPlaneTrackDir(hpoint1, trackMom, point2) > dR) isIsolated = 1;
    else isIsolated = 0;
    return isIsolated;
  } 

}
