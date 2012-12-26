#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalEndcapGeometry.h"

#include "Calibration/IsolatedParticles/interface/CaloPropagateTrack.h"

#include <iostream>

namespace spr{

  std::vector<spr::propagatedTrackID> propagateCALO(edm::Handle<reco::TrackCollection>& trkCollection, const CaloGeometry* geo, const MagneticField* bField, std::string & theTrackQuality, bool debug) {

    std::vector<spr::propagatedTrackID> vdets;
    spr::propagateCALO(trkCollection,geo,bField,theTrackQuality, vdets, debug);
    return vdets;
  }

  void propagateCALO(edm::Handle<reco::TrackCollection>& trkCollection, const CaloGeometry* geo, const MagneticField* bField, std::string & theTrackQuality, std::vector<spr::propagatedTrackID>& vdets, bool debug) {

    const EcalBarrelGeometry *barrelGeom = (dynamic_cast< const EcalBarrelGeometry *> (geo->getSubdetectorGeometry(DetId::Ecal,EcalBarrel)));
    const EcalEndcapGeometry *endcapGeom = (dynamic_cast< const EcalEndcapGeometry *> (geo->getSubdetectorGeometry(DetId::Ecal,EcalEndcap)));
    const CaloSubdetectorGeometry* gHB = geo->getSubdetectorGeometry(DetId::Hcal,HcalBarrel);
    reco::TrackBase::TrackQuality trackQuality_=reco::TrackBase::qualityByName(theTrackQuality);

    unsigned int indx;
    reco::TrackCollection::const_iterator trkItr;
    for (trkItr = trkCollection->begin(),indx=0; trkItr != trkCollection->end(); ++trkItr,indx++) {
      const reco::Track* pTrack = &(*trkItr);
      spr::propagatedTrackID vdet;
      vdet.trkItr = trkItr;
      vdet.ok     = (pTrack->quality(trackQuality_));
      vdet.detIdECAL = DetId(0);
      vdet.detIdHCAL = DetId(0);
      vdet.detIdEHCAL= DetId(0);
      if (debug) std::cout << "Propagate track " << indx << " p " << trkItr->p() << " eta " << trkItr->eta() << " phi " << trkItr->phi() << " Flag " << vdet.ok << std::endl;

      std::pair<math::XYZPoint,bool> info = spr::propagateECAL (pTrack, bField, debug);
      vdet.okECAL = info.second;
      if (vdet.okECAL) {
	const GlobalPoint point(info.first.x(),info.first.y(),info.first.z());
	vdet.etaECAL = point.eta();
	vdet.phiECAL = point.phi();
	if (std::abs(point.eta())<1.479) {
	  vdet.detIdECAL = barrelGeom->getClosestCell(point);
	} else {
	  vdet.detIdECAL = endcapGeom->getClosestCell(point);
	}
	vdet.detIdEHCAL = gHB->getClosestCell(point);
      }
      info = spr::propagateHCAL (pTrack, bField, debug);
      vdet.okHCAL = info.second;
      if (vdet.okHCAL) {
	const GlobalPoint point(info.first.x(),info.first.y(),info.first.z());
	vdet.etaHCAL = point.eta();
	vdet.phiHCAL = point.phi();
	vdet.detIdHCAL = gHB->getClosestCell(point);
      }

      vdets.push_back(vdet);
    }
    
    if (debug) {
      std::cout << "propagateCALO:: for " << vdets.size() << " tracks" << std::endl;
      for (unsigned int i=0; i<vdets.size(); ++i) {
	std::cout << "Track [" << i << "] Flag: " << vdets[i].ok << " ECAL (" << vdets[i].okECAL << ") ";
	if (vdets[i].detIdECAL.subdetId() == EcalBarrel) {
	  std::cout << (EBDetId)(vdets[i].detIdECAL);
	} else {
	  std::cout << (EEDetId)(vdets[i].detIdECAL); 
	}
	std::cout << " HCAL (" << vdets[i].okHCAL << ") " << (HcalDetId)(vdets[i].detIdHCAL) << " Or " << (HcalDetId)(vdets[i].detIdEHCAL) << std::endl;
      }
    }
  }

  void propagateCALO(edm::Handle<reco::TrackCollection>& trkCollection, const CaloGeometry* geo, const MagneticField* bField, std::string & theTrackQuality, std::vector<spr::propagatedTrackDirection>& trkDir, bool debug) {

    const EcalBarrelGeometry *barrelGeom = (dynamic_cast< const EcalBarrelGeometry *> (geo->getSubdetectorGeometry(DetId::Ecal,EcalBarrel)));
    const EcalEndcapGeometry *endcapGeom = (dynamic_cast< const EcalEndcapGeometry *> (geo->getSubdetectorGeometry(DetId::Ecal,EcalEndcap)));
    const CaloSubdetectorGeometry* gHB = geo->getSubdetectorGeometry(DetId::Hcal,HcalBarrel);
    reco::TrackBase::TrackQuality trackQuality_=reco::TrackBase::qualityByName(theTrackQuality);

    unsigned int indx;
    reco::TrackCollection::const_iterator trkItr;
    for (trkItr = trkCollection->begin(),indx=0; trkItr != trkCollection->end(); ++trkItr,indx++) {
      const reco::Track* pTrack = &(*trkItr);
      spr::propagatedTrackDirection trkD;
      trkD.trkItr = trkItr;
      trkD.ok     = (pTrack->quality(trackQuality_));
      trkD.detIdECAL = DetId(0);
      trkD.detIdHCAL = DetId(0);
      trkD.detIdEHCAL= DetId(0);
      if (debug) std::cout << "Propagate track " << indx << " p " << trkItr->p() << " eta " << trkItr->eta() << " phi " << trkItr->phi() << " Flag " << trkD.ok << std::endl;

      spr::propagatedTrack info = spr::propagateTrackToECAL (pTrack, bField, debug);
      GlobalPoint point(info.point.x(),info.point.y(),info.point.z());
      trkD.okECAL        = info.ok;
      trkD.pointECAL     = point;
      trkD.directionECAL = info.direction;
      if (trkD.okECAL) {
	if (std::abs(info.point.eta())<1.479) {
	  trkD.detIdECAL = barrelGeom->getClosestCell(point);
	} else {
	  trkD.detIdECAL = endcapGeom->getClosestCell(point);
	}
	trkD.detIdEHCAL = gHB->getClosestCell(point);
      }
      info = spr::propagateTrackToHCAL (pTrack, bField, debug);
      point = GlobalPoint(info.point.x(),info.point.y(),info.point.z());
      trkD.okHCAL        = info.ok;
      trkD.pointHCAL     = point;
      trkD.directionHCAL = info.direction;
      if (trkD.okHCAL) {
	trkD.detIdHCAL = gHB->getClosestCell(point);
      }
      trkDir.push_back(trkD);
    }
    
    if (debug) {
      std::cout << "propagateCALO:: for " << trkDir.size() << " tracks" << std::endl;
      for (unsigned int i=0; i<trkDir.size(); ++i) {
	std::cout << "Track [" << i << "] Flag: " << trkDir[i].ok << " ECAL (" << trkDir[i].okECAL << ")";
	if (trkDir[i].okECAL) {
	  std::cout << " point " << trkDir[i].pointECAL << " direction "
		    << trkDir[i].directionECAL << " "; 
	  if (trkDir[i].detIdECAL.subdetId() == EcalBarrel) {
	    std::cout << (EBDetId)(trkDir[i].detIdECAL);
	  } else {
	    std::cout << (EEDetId)(trkDir[i].detIdECAL); 
	  }
	}
	std::cout << " HCAL (" << trkDir[i].okHCAL << ")";
	if (trkDir[i].okHCAL) {
	  std::cout << " point " << trkDir[i].pointHCAL << " direction "
		    << trkDir[i].directionHCAL << " " 
		    << (HcalDetId)(trkDir[i].detIdHCAL); 
	}
	std::cout << " Or " << (HcalDetId)(trkDir[i].detIdEHCAL) << std::endl;
      }
    }
  }

  std::vector<spr::propagatedGenTrackID> propagateCALO(const HepMC::GenEvent * genEvent, edm::ESHandle<ParticleDataTable>& pdt, const CaloGeometry* geo, const MagneticField* bField, double etaMax, bool debug) {

    const EcalBarrelGeometry *barrelGeom = (dynamic_cast< const EcalBarrelGeometry *> (geo->getSubdetectorGeometry(DetId::Ecal,EcalBarrel)));
    const EcalEndcapGeometry *endcapGeom = (dynamic_cast< const EcalEndcapGeometry *> (geo->getSubdetectorGeometry(DetId::Ecal,EcalEndcap)));
    const CaloSubdetectorGeometry* gHB = geo->getSubdetectorGeometry(DetId::Hcal,HcalBarrel);

    std::vector<spr::propagatedGenTrackID> trkDir;
    unsigned int indx;
    HepMC::GenEvent::particle_const_iterator p;
    for (p=genEvent->particles_begin(),indx=0;   p != genEvent->particles_end(); ++p,++indx) {
      spr::propagatedGenTrackID trkD;
      trkD.trkItr    = p;
      trkD.detIdECAL = DetId(0);
      trkD.detIdHCAL = DetId(0);
      trkD.detIdEHCAL= DetId(0);
      trkD.pdgId  = ((*p)->pdg_id());
      trkD.charge = ((pdt->particle(trkD.pdgId))->ID().threeCharge())/3;
      GlobalVector momentum = GlobalVector((*p)->momentum().px(), (*p)->momentum().py(), (*p)->momentum().pz());
      if (debug) std::cout << "Propagate track " << indx << " pdg " << trkD.pdgId << " charge " << trkD.charge << " p " << momentum << std::endl;
      
      // consider stable particles
      if ( (*p)->status()==1 && std::abs((*p)->momentum().eta()) < etaMax ) { 
	GlobalPoint vertex = GlobalPoint(0.1*(*p)->production_vertex()->position().x(), 
					 0.1*(*p)->production_vertex()->position().y(), 
					 0.1*(*p)->production_vertex()->position().z());
	trkD.ok = true;
	spr::propagatedTrack info = spr::propagateCalo (vertex, momentum, trkD.charge, bField, 319.2, 129.4, 1.479, debug);
	GlobalPoint point(info.point.x(),info.point.y(),info.point.z());
	trkD.okECAL        = info.ok;
	trkD.pointECAL     = point;
	trkD.directionECAL = info.direction;
	if (trkD.okECAL) {
	  if (std::abs(info.point.eta())<1.479) {
	    trkD.detIdECAL = barrelGeom->getClosestCell(point);
	  } else {
	    trkD.detIdECAL = endcapGeom->getClosestCell(point);
	  }
	  trkD.detIdEHCAL = gHB->getClosestCell(point);
	}

	info = spr::propagateCalo (vertex, momentum, trkD.charge, bField, 402.7, 180.7, 1.392, debug);
	point = GlobalPoint(info.point.x(),info.point.y(),info.point.z());
	trkD.okHCAL        = info.ok;
	trkD.pointHCAL     = point;
	trkD.directionHCAL = info.direction;
	if (trkD.okHCAL) {
	  trkD.detIdHCAL = gHB->getClosestCell(point);
	}
      }
      trkDir.push_back(trkD);
    }

    if (debug) {
      std::cout << "propagateCALO:: for " << trkDir.size() << " tracks" << std::endl;
      for (unsigned int i=0; i<trkDir.size(); ++i) {
	if (trkDir[i].okECAL) std::cout << "Track [" << i << "] Flag: " << trkDir[i].ok << " ECAL (" << trkDir[i].okECAL << ")";
	if (trkDir[i].okECAL) {
	  std::cout << " point " << trkDir[i].pointECAL << " direction "
		    << trkDir[i].directionECAL << " "; 
	  if (trkDir[i].detIdECAL.subdetId() == EcalBarrel) {
	    std::cout << (EBDetId)(trkDir[i].detIdECAL);
	  } else {
	    std::cout << (EEDetId)(trkDir[i].detIdECAL); 
	  }
	}
	if (trkDir[i].okECAL) std::cout << " HCAL (" << trkDir[i].okHCAL << ")";
	if (trkDir[i].okHCAL) {
	  std::cout << " point " << trkDir[i].pointHCAL << " direction "
		    << trkDir[i].directionHCAL << " " 
		    << (HcalDetId)(trkDir[i].detIdHCAL); 
	}
	if (trkDir[i].okECAL) std::cout << " Or " << (HcalDetId)(trkDir[i].detIdEHCAL) << std::endl;
      }
    }
    return trkDir;
  }

  std::vector<spr::propagatedGenParticleID> propagateCALO(edm::Handle<reco::GenParticleCollection>& genParticles, edm::ESHandle<ParticleDataTable>& pdt, const CaloGeometry* geo, const MagneticField* bField, double etaMax, bool debug) {

    const EcalBarrelGeometry *barrelGeom = (dynamic_cast< const EcalBarrelGeometry *> (geo->getSubdetectorGeometry(DetId::Ecal,EcalBarrel)));
    const EcalEndcapGeometry *endcapGeom = (dynamic_cast< const EcalEndcapGeometry *> (geo->getSubdetectorGeometry(DetId::Ecal,EcalEndcap)));
    const CaloSubdetectorGeometry* gHB = geo->getSubdetectorGeometry(DetId::Hcal,HcalBarrel);

    std::vector<spr::propagatedGenParticleID> trkDir;
    unsigned int indx;
    reco::GenParticleCollection::const_iterator p;
    for (p=genParticles->begin(),indx=0;   p != genParticles->end(); ++p,++indx) {
      spr::propagatedGenParticleID trkD;
      trkD.trkItr    = p;
      trkD.detIdECAL = DetId(0);
      trkD.detIdHCAL = DetId(0);
      trkD.detIdEHCAL= DetId(0);
      trkD.pdgId     = (p->pdgId());
      trkD.charge    = p->charge();
      GlobalVector momentum = GlobalVector(p->momentum().x(), p->momentum().y(), p->momentum().z());
      if (debug) std::cout << "Propagate track " << indx << " pdg " << trkD.pdgId << " charge " << trkD.charge << " p " << momentum << std::endl;
      
      // consider stable particles
      if ( p->status()==1 && std::abs(momentum.eta()) < etaMax ) { 
	GlobalPoint vertex = GlobalPoint(p->vertex().x(), p->vertex().y(), p->vertex().z());
	trkD.ok = true;
	spr::propagatedTrack info = spr::propagateCalo (vertex, momentum, trkD.charge, bField, 319.2, 129.4, 1.479, debug);
	GlobalPoint point(info.point.x(),info.point.y(),info.point.z());
	trkD.okECAL        = info.ok;
	trkD.pointECAL     = point;
	trkD.directionECAL = info.direction;
	if (trkD.okECAL) {
	  if (std::abs(info.point.eta())<1.479) {
	    trkD.detIdECAL = barrelGeom->getClosestCell(point);
	  } else {
	    trkD.detIdECAL = endcapGeom->getClosestCell(point);
	  }
	  trkD.detIdEHCAL = gHB->getClosestCell(point);
	}

	info = spr::propagateCalo (vertex, momentum, trkD.charge, bField, 402.7, 180.7, 1.392, debug);
	point = GlobalPoint(info.point.x(),info.point.y(),info.point.z());
	trkD.okHCAL        = info.ok;
	trkD.pointHCAL     = point;
	trkD.directionHCAL = info.direction;
	if (trkD.okHCAL) {
	  trkD.detIdHCAL = gHB->getClosestCell(point);
	}
      }
      trkDir.push_back(trkD);
    }

    if (debug) {
      std::cout << "propagateCALO:: for " << trkDir.size() << " tracks" << std::endl;
      for (unsigned int i=0; i<trkDir.size(); ++i) {
	if (trkDir[i].okECAL) std::cout << "Track [" << i << "] Flag: " << trkDir[i].ok << " ECAL (" << trkDir[i].okECAL << ")";
	if (trkDir[i].okECAL) {
	  std::cout << " point " << trkDir[i].pointECAL << " direction "
		    << trkDir[i].directionECAL << " "; 
	  if (trkDir[i].detIdECAL.subdetId() == EcalBarrel) {
	    std::cout << (EBDetId)(trkDir[i].detIdECAL);
	  } else {
	    std::cout << (EEDetId)(trkDir[i].detIdECAL); 
	  }
	}
	if (trkDir[i].okECAL) std::cout << " HCAL (" << trkDir[i].okHCAL << ")";
	if (trkDir[i].okHCAL) {
	  std::cout << " point " << trkDir[i].pointHCAL << " direction "
		    << trkDir[i].directionHCAL << " " 
		    << (HcalDetId)(trkDir[i].detIdHCAL); 
	}
	if (trkDir[i].okECAL) std::cout << " Or " << (HcalDetId)(trkDir[i].detIdEHCAL) << std::endl;
      }
    }
    return trkDir;
  }

  spr::propagatedTrackDirection propagateCALO(unsigned int thisTrk, edm::Handle<edm::SimTrackContainer>& SimTk, edm::Handle<edm::SimVertexContainer>& SimVtx, const CaloGeometry* geo, const MagneticField* bField, bool debug) {

    const EcalBarrelGeometry *barrelGeom = (dynamic_cast< const EcalBarrelGeometry *> (geo->getSubdetectorGeometry(DetId::Ecal,EcalBarrel)));
    const EcalEndcapGeometry *endcapGeom = (dynamic_cast< const EcalEndcapGeometry *> (geo->getSubdetectorGeometry(DetId::Ecal,EcalEndcap)));
    const CaloSubdetectorGeometry* gHB = geo->getSubdetectorGeometry(DetId::Hcal,HcalBarrel);

    spr::trackAtOrigin   trk = spr::simTrackAtOrigin(thisTrk, SimTk, SimVtx, debug);
    spr::propagatedTrackDirection trkD;
    trkD.ok     = trk.ok;
    trkD.detIdECAL = DetId(0);
    trkD.detIdHCAL = DetId(0);
    trkD.detIdEHCAL= DetId(0);
    if (debug) std::cout << "Propagate track " << thisTrk << " charge " << trk.charge << " position " << trk.position << " p " << trk.momentum << " Flag " << trkD.ok << std::endl;

    if (trkD.ok) {
      spr::propagatedTrack info = spr::propagateCalo (trk.position, trk.momentum, trk.charge, bField, 319.2, 129.4, 1.479, debug);
      GlobalPoint point(info.point.x(),info.point.y(),info.point.z());
      trkD.okECAL        = info.ok;
      trkD.pointECAL     = point;
      trkD.directionECAL = info.direction;
      if (trkD.okECAL) {
	if (std::abs(info.point.eta())<1.479) {
	  trkD.detIdECAL = barrelGeom->getClosestCell(point);
	} else {
	  trkD.detIdECAL = endcapGeom->getClosestCell(point);
	}
	trkD.detIdEHCAL = gHB->getClosestCell(point);
      }

      info = spr::propagateCalo (trk.position, trk.momentum, trk.charge, bField, 402.7, 180.7, 1.392, debug);
      point = GlobalPoint(info.point.x(),info.point.y(),info.point.z());
      trkD.okHCAL        = info.ok;
      trkD.pointHCAL     = point;
      trkD.directionHCAL = info.direction;
      if (trkD.okHCAL) {
	trkD.detIdHCAL = gHB->getClosestCell(point);
      }
    }

    if (debug) {
      std::cout << "propagateCALO:: for track [" << thisTrk << "] Flag: " << trkD.ok << " ECAL (" << trkD.okECAL << ") HCAL (" << trkD.okHCAL << ")" << std::endl;
      if (trkD.okECAL) {
	std::cout << "ECAL point " << trkD.pointECAL << " direction "
		  << trkD.directionECAL << " "; 
	if (trkD.detIdECAL.subdetId() == EcalBarrel) {
	  std::cout << (EBDetId)(trkD.detIdECAL);
	} else {
	  std::cout << (EEDetId)(trkD.detIdECAL); 
	}
      }
      if (trkD.okHCAL) {
	std::cout << " HCAL point " << trkD.pointHCAL << " direction "
		  << trkD.directionHCAL << " " << (HcalDetId)(trkD.detIdHCAL); 
      }
      if (trkD.okECAL) std::cout << " Or " << (HcalDetId)(trkD.detIdEHCAL);
      std::cout << std::endl;
    }

    return trkD;
  }

  propagatedTrack propagateTrackToECAL(const reco::Track *track, const MagneticField* bfield, bool debug) {
    GlobalPoint  vertex (track->vx(), track->vy(), track->vz());
    GlobalVector momentum (track->px(), track->py(), track->pz());
    int charge (track->charge());
    return spr::propagateCalo (vertex, momentum, charge, bfield, 319.2, 129.4, 1.479, debug);
  }

  propagatedTrack propagateTrackToECAL(unsigned int thisTrk, edm::Handle<edm::SimTrackContainer>& SimTk, edm::Handle<edm::SimVertexContainer>& SimVtx, const MagneticField* bfield, bool debug) {

    spr::trackAtOrigin   trk = spr::simTrackAtOrigin(thisTrk, SimTk, SimVtx, debug);
    spr::propagatedTrack ptrk;
    if (trk.ok) 
      ptrk = spr::propagateCalo (trk.position, trk.momentum, trk.charge, bfield, 319.2, 129.4, 1.479, debug);
    return ptrk;
  }

  std::pair<math::XYZPoint,bool> propagateECAL(const reco::Track *track, const MagneticField* bfield, bool debug) {    
    GlobalPoint  vertex (track->vx(), track->vy(), track->vz());
    GlobalVector momentum (track->px(), track->py(), track->pz());
    int charge (track->charge());
    return spr::propagateECAL (vertex, momentum, charge, bfield, debug);
  }

  std::pair<math::XYZPoint,bool> propagateECAL(const GlobalPoint& vertex, const GlobalVector& momentum, int charge, const MagneticField* bfield, bool debug) {
    spr::propagatedTrack track = spr::propagateCalo (vertex, momentum, charge, bfield, 319.2, 129.4, 1.479, debug);
    return std::pair<math::XYZPoint,bool>(track.point,track.ok);
  }

  spr::propagatedTrack propagateTrackToHCAL(const reco::Track *track, const MagneticField* bfield, bool debug) {
    GlobalPoint  vertex (track->vx(), track->vy(), track->vz());
    GlobalVector momentum (track->px(), track->py(), track->pz());
    int charge (track->charge());
    return spr::propagateCalo (vertex, momentum, charge, bfield, 402.7, 180.7, 1.392, debug);
  }

  spr::propagatedTrack propagateTrackToHCAL(unsigned int thisTrk, edm::Handle<edm::SimTrackContainer>& SimTk, edm::Handle<edm::SimVertexContainer>& SimVtx, const MagneticField* bfield, bool debug) {
    spr::trackAtOrigin   trk = spr::simTrackAtOrigin(thisTrk, SimTk, SimVtx, debug);
    spr::propagatedTrack ptrk;
    if (trk.ok) 
      ptrk = spr::propagateCalo (trk.position, trk.momentum, trk.charge, bfield, 402.7, 180.7, 1.392, debug);
    return ptrk;
  }

  std::pair<math::XYZPoint,bool> propagateHCAL(const reco::Track *track, const MagneticField* bfield, bool debug) {
    GlobalPoint  vertex (track->vx(), track->vy(), track->vz());
    GlobalVector momentum (track->px(), track->py(), track->pz());
    int charge (track->charge());
    return spr::propagateHCAL (vertex, momentum, charge, bfield, debug);
  }

  std::pair<math::XYZPoint,bool> propagateHCAL(const GlobalPoint& vertex, const GlobalVector& momentum, int charge, const MagneticField* bfield, bool debug) {
    spr::propagatedTrack track = spr::propagateCalo (vertex, momentum, charge, bfield, 402.7, 180.7, 1.392, debug);
    return std::pair<math::XYZPoint,bool>(track.point,track.ok);
  }

  std::pair<math::XYZPoint,bool> propagateTracker(const reco::Track *track, const MagneticField* bfield, bool debug) {
    GlobalPoint  vertex (track->vx(), track->vy(), track->vz());
    GlobalVector momentum (track->px(), track->py(), track->pz());
    int charge (track->charge());
    spr::propagatedTrack track1 = spr::propagateCalo (vertex, momentum, charge, bfield, 290.0, 109.0, 1.705, debug);
    return std::pair<math::XYZPoint,bool>(track1.point,track1.ok);
  }

  std::pair<math::XYZPoint,double> propagateTrackerEnd(const reco::Track *track, const MagneticField* bField, bool debug) {

    GlobalPoint  vertex (track->vx(), track->vy(), track->vz());
    GlobalVector momentum (track->px(), track->py(), track->pz());
    int charge (track->charge());
    float radius = track->outerPosition().Rho();
    float zdist  = track->outerPosition().Z();
    if (debug) std::cout << "propagateTrackerEnd:: Vertex " << vertex << " Momentum " << momentum << " Charge " << charge << " Radius " << radius << " Z " << zdist << std::endl;

    FreeTrajectoryState fts (vertex, momentum, charge, bField);
    Plane::PlanePointer endcap = Plane::build(Plane::PositionType (0, 0, zdist), Plane::RotationType());
    Cylinder::CylinderPointer barrel = Cylinder::build(radius, Cylinder::PositionType (0, 0, 0), Cylinder::RotationType ());

    AnalyticalPropagator myAP (bField, alongMomentum, 2*M_PI);

    TrajectoryStateOnSurface tsose = myAP.propagate(fts, *endcap);
    TrajectoryStateOnSurface tsosb = myAP.propagate(fts, *barrel);

    math::XYZPoint point(-999.,-999.,-999.);
    bool ok=false;
    GlobalVector direction(0,0,1);
    if (tsosb.isValid() && std::abs(zdist) < 110) {
      point.SetXYZ(tsosb.globalPosition().x(), tsosb.globalPosition().y(), tsosb.globalPosition().z());
      direction = tsosb.globalDirection();
      ok = true;
    } else if (tsose.isValid()) {
      point.SetXYZ(tsose.globalPosition().x(), tsose.globalPosition().y(), tsose.globalPosition().z());
      direction = tsose.globalDirection();
      ok = true;
    }

    double length = -1;
    if (ok) {
      math::XYZPoint vDiff(point.x()-vertex.x(), point.y()-vertex.y(), point.z()-vertex.z());
      double dphi  = direction.phi()-momentum.phi();
      double rdist = std::sqrt(vDiff.x()*vDiff.x()+vDiff.y()*vDiff.y());
      double rat   = 0.5*dphi/std::sin(0.5*dphi);
      double dZ    = vDiff.z();
      double dS    = rdist*rat; //dZ*momentum.z()/momentum.perp();
      length       = std::sqrt(dS*dS+dZ*dZ);
      if (debug) 
	std::cout << "propagateTracker:: Barrel " << tsosb.isValid() << " Endcap " << tsose.isValid() << " OverAll " << ok << " Point " << point << " RDist " << rdist << " dS " << dS << " dS/pt " << rdist*rat/momentum.perp() << " zdist " << dZ << " dz/pz " << dZ/momentum.z() << " Length " << length << std::endl;
    }

    return std::pair<math::XYZPoint,double>(point,length);
  }

  spr::propagatedTrack propagateCalo(const GlobalPoint& tpVertex, const GlobalVector& tpMomentum, int tpCharge, const MagneticField* bField, float zdist, float radius, float corner, bool debug) {
    
    spr::propagatedTrack track;
    if (debug) std::cout << "propagateCalo:: Vertex " << tpVertex << " Momentum " << tpMomentum << " Charge " << tpCharge << " Radius " << radius << " Z " << zdist << " Corner " << corner << std::endl;
    FreeTrajectoryState fts (tpVertex, tpMomentum, tpCharge, bField);
    
    Plane::PlanePointer lendcap = Plane::build(Plane::PositionType (0, 0, -zdist), Plane::RotationType());
    Plane::PlanePointer rendcap = Plane::build(Plane::PositionType (0, 0,  zdist), Plane::RotationType());
    
    Cylinder::CylinderPointer barrel = Cylinder::build(radius, Cylinder::PositionType (0, 0, 0), Cylinder::RotationType ());
  
    AnalyticalPropagator myAP (bField, alongMomentum, 2*M_PI);

    TrajectoryStateOnSurface tsose;
    if (tpMomentum.eta() < 0) {
      tsose = myAP.propagate(fts, *lendcap);
    } else {
      tsose = myAP.propagate(fts, *rendcap);
    }

    TrajectoryStateOnSurface tsosb = myAP.propagate(fts, *barrel);

    track.ok=true;
    if (tsose.isValid() && tsosb.isValid()) {
      float absEta = std::abs(tsosb.globalPosition().eta());
      if (absEta < corner) {
	track.point.SetXYZ(tsosb.globalPosition().x(), tsosb.globalPosition().y(), tsosb.globalPosition().z());
	track.direction = tsosb.globalDirection();
      } else {
	track.point.SetXYZ(tsose.globalPosition().x(), tsose.globalPosition().y(), tsose.globalPosition().z());
	track.direction = tsose.globalDirection();
      }
    } else if (tsose.isValid()) {
      track.point.SetXYZ(tsose.globalPosition().x(), tsose.globalPosition().y(), tsose.globalPosition().z());
      track.direction = tsose.globalDirection();
    } else if (tsosb.isValid()) {
      track.point.SetXYZ(tsosb.globalPosition().x(), tsosb.globalPosition().y(), tsosb.globalPosition().z());
      track.direction = tsosb.globalDirection();
    } else {
      track.point.SetXYZ(-999., -999., -999.);
      track.direction = GlobalVector(0,0,1);
      track.ok = false;
    }
    if (debug) {
      std::cout << "propagateCalo:: Barrel " << tsosb.isValid() << " Endcap " << tsose.isValid() << " OverAll " << track.ok << " Point " << track.point << " Direction " << track.direction << std::endl;
      if (track.ok) {
	math::XYZPoint vDiff(track.point.x()-tpVertex.x(), track.point.y()-tpVertex.y(), track.point.z()-tpVertex.z());
	double dphi = track.direction.phi()-tpMomentum.phi();
	double rdist = std::sqrt(vDiff.x()*vDiff.x()+vDiff.y()*vDiff.y());
	double pt    = tpMomentum.perp();
	double rat   = 0.5*dphi/std::sin(0.5*dphi);
	std::cout << "RDist " << rdist << " pt " << pt << " r/pt " << rdist*rat/pt << " zdist " << vDiff.z() << " pz " << tpMomentum.z() << " z/pz " << vDiff.z()/tpMomentum.z() << std::endl;
      }
    }
    return track;
  }

  spr::trackAtOrigin simTrackAtOrigin(unsigned int thisTrk, edm::Handle<edm::SimTrackContainer>& SimTk, edm::Handle<edm::SimVertexContainer>& SimVtx, bool debug) {

    spr::trackAtOrigin trk;

    edm::SimTrackContainer::const_iterator itr = SimTk->end();
    for (edm::SimTrackContainer::const_iterator simTrkItr = SimTk->begin(); simTrkItr!= SimTk->end(); simTrkItr++) {
      if ( simTrkItr->trackId() == thisTrk ) {
	if (debug) std::cout << "matched trackId (maximum occurance) " << thisTrk << " type " << simTrkItr->type() << std::endl;
	itr = simTrkItr;
	break;
      }
    }

    if (itr != SimTk->end()) {
      int vertIndex = itr->vertIndex();
      if (vertIndex != -1 && vertIndex < (int)SimVtx->size()) {
	edm::SimVertexContainer::const_iterator simVtxItr= SimVtx->begin();
	for (int iv=0; iv<vertIndex; iv++) simVtxItr++;
	const math::XYZTLorentzVectorD pos = simVtxItr->position();
	const math::XYZTLorentzVectorD mom = itr->momentum();
	trk.ok = true;
	trk.charge   = (int)(itr->charge());
	trk.position = GlobalPoint(pos.x(), pos.y(), pos.z());
	trk.momentum = GlobalVector(mom.x(), mom.y(), mom.z());
      }
    }
    if (debug) std::cout << "Track flag " << trk.ok << " Position " << trk.position << " Momentum " << trk.momentum << std::endl;;
    return trk;
  }

}
