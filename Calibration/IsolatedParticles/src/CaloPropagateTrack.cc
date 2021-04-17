#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"

#include "Calibration/IsolatedParticles/interface/CaloConstants.h"
#include "Calibration/IsolatedParticles/interface/CaloPropagateTrack.h"

#include <iostream>

//#define EDM_ML_DEBUG

namespace spr {

  std::vector<spr::propagatedTrackID> propagateCosmicCALO(edm::Handle<reco::TrackCollection>& trkCollection,
                                                          const CaloGeometry* geo,
                                                          const MagneticField* bField,
                                                          const std::string& theTrackQuality,
                                                          bool debug) {
    const CaloSubdetectorGeometry* barrelGeom = geo->getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
    const CaloSubdetectorGeometry* endcapGeom = geo->getSubdetectorGeometry(DetId::Ecal, EcalEndcap);
    const CaloSubdetectorGeometry* gHB = geo->getSubdetectorGeometry(DetId::Hcal, HcalBarrel);
    reco::TrackBase::TrackQuality trackQuality_ = reco::TrackBase::qualityByName(theTrackQuality);
    std::vector<spr::propagatedTrackID> vdets;

    unsigned int indx;
    reco::TrackCollection::const_iterator trkItr;
    for (trkItr = trkCollection->begin(), indx = 0; trkItr != trkCollection->end(); ++trkItr, indx++) {
      const reco::Track* pTrack = &(*trkItr);
      spr::propagatedTrackID vdet;
      vdet.trkItr = trkItr;
      vdet.ok = (trackQuality_ != reco::TrackBase::undefQuality) ? (pTrack->quality(trackQuality_)) : true;
      vdet.detIdECAL = DetId(0);
      vdet.detIdHCAL = DetId(0);
      vdet.detIdEHCAL = DetId(0);
#ifdef EDM_ML_DEBUG
      if (debug)
        std::cout << "Propagate track " << indx << " p " << trkItr->p() << " eta " << trkItr->eta() << " phi "
                  << trkItr->phi() << " Flag " << vdet.ok << std::endl;
#endif
      GlobalPoint vertex;
      GlobalVector momentum;
      int charge(pTrack->charge());
      if (((pTrack->innerPosition()).Perp2()) < ((pTrack->outerPosition()).Perp2())) {
        vertex = GlobalPoint(
            ((pTrack->innerPosition()).X()), ((pTrack->innerPosition()).Y()), ((pTrack->innerPosition()).Z()));
        momentum = GlobalVector(
            ((pTrack->innerMomentum()).X()), ((pTrack->innerMomentum()).Y()), ((pTrack->innerMomentum()).Z()));
      } else {
        vertex = GlobalPoint(
            ((pTrack->outerPosition()).X()), ((pTrack->outerPosition()).Y()), ((pTrack->outerPosition()).Z()));
        momentum = GlobalVector(
            ((pTrack->outerMomentum()).X()), ((pTrack->outerMomentum()).Y()), ((pTrack->outerMomentum()).Z()));
      }
#ifdef EDM_ML_DEBUG
      if (debug)
        std::cout << "Track charge " << charge << " p " << momentum << " position " << vertex << std::endl;
#endif
      std::pair<math::XYZPoint, bool> info = spr::propagateECAL(vertex, momentum, charge, bField, debug);
#ifdef EDM_ML_DEBUG
      if (debug)
        std::cout << "Propagate to ECAL " << info.second << " at (" << info.first.x() << ", " << info.first.y() << ", "
                  << info.first.z() << ")\n";
#endif

      vdet.okECAL = info.second;
      if (vdet.okECAL) {
        const GlobalPoint point(info.first.x(), info.first.y(), info.first.z());
        vdet.etaECAL = point.eta();
        vdet.phiECAL = point.phi();
        if (std::abs(point.eta()) < spr::etaBEEcal) {
          vdet.detIdECAL = barrelGeom->getClosestCell(point);
        } else {
          if (endcapGeom)
            vdet.detIdECAL = endcapGeom->getClosestCell(point);
          else
            vdet.okECAL = false;
        }
        vdet.detIdEHCAL = gHB->getClosestCell(point);
#ifdef EDM_ML_DEBUG
        if (debug) {
          std::cout << "Point at ECAL (" << vdet.etaECAL << ", " << vdet.phiECAL << " ";
          if (std::abs(point.eta()) < spr::etaBEEcal)
            std::cout << EBDetId(vdet.detIdECAL);
          else
            std::cout << EEDetId(vdet.detIdECAL);
          std::cout << " " << HcalDetId(vdet.detIdEHCAL) << std::endl;
        }
#endif
      }
      info = spr::propagateHCAL(vertex, momentum, charge, bField, debug);
#ifdef EDM_ML_DEBUG
      if (debug)
        std::cout << "Propagate to HCAL " << info.second << " at (" << info.first.x() << ", " << info.first.y() << ", "
                  << info.first.z() << ")\n";
#endif
      vdet.okHCAL = info.second;
      if (vdet.okHCAL) {
        const GlobalPoint point(info.first.x(), info.first.y(), info.first.z());
        vdet.etaHCAL = point.eta();
        vdet.phiHCAL = point.phi();
        vdet.detIdHCAL = gHB->getClosestCell(point);
      }
#ifdef EDM_ML_DEBUG
      if (debug) {
        std::cout << "Track [" << indx << "] Flag: " << vdet.ok << " ECAL (" << vdet.okECAL << ") ";
        if (vdet.detIdECAL.subdetId() == EcalBarrel)
          std::cout << (EBDetId)(vdet.detIdECAL);
        else
          std::cout << (EEDetId)(vdet.detIdECAL);
        std::cout << " HCAL (" << vdet.okHCAL << ") " << (HcalDetId)(vdet.detIdHCAL) << " Or "
                  << (HcalDetId)(vdet.detIdEHCAL) << std::endl;
      }
#endif
      vdets.push_back(vdet);
    }

#ifdef EDM_ML_DEBUG
    if (debug) {
      std::cout << "propagateCALO:: for " << vdets.size() << " tracks" << std::endl;
      for (unsigned int i = 0; i < vdets.size(); ++i) {
        std::cout << "Track [" << i << "] Flag: " << vdets[i].ok << " ECAL (" << vdets[i].okECAL << ") ";
        if (vdets[i].detIdECAL.subdetId() == EcalBarrel) {
          std::cout << (EBDetId)(vdets[i].detIdECAL);
        } else {
          std::cout << (EEDetId)(vdets[i].detIdECAL);
        }
        std::cout << " HCAL (" << vdets[i].okHCAL << ") " << (HcalDetId)(vdets[i].detIdHCAL) << " Or "
                  << (HcalDetId)(vdets[i].detIdEHCAL) << std::endl;
      }
    }
#endif
    return vdets;
  }

  std::vector<spr::propagatedTrackID> propagateCALO(edm::Handle<reco::TrackCollection>& trkCollection,
                                                    const CaloGeometry* geo,
                                                    const MagneticField* bField,
                                                    const std::string& theTrackQuality,
                                                    bool debug) {
    std::vector<spr::propagatedTrackID> vdets;
    spr::propagateCALO(trkCollection, geo, bField, theTrackQuality, vdets, debug);
    return vdets;
  }

  void propagateCALO(edm::Handle<reco::TrackCollection>& trkCollection,
                     const CaloGeometry* geo,
                     const MagneticField* bField,
                     const std::string& theTrackQuality,
                     std::vector<spr::propagatedTrackID>& vdets,
                     bool debug) {
    const CaloSubdetectorGeometry* barrelGeom = geo->getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
    const CaloSubdetectorGeometry* endcapGeom = geo->getSubdetectorGeometry(DetId::Ecal, EcalEndcap);
    const CaloSubdetectorGeometry* gHB = geo->getSubdetectorGeometry(DetId::Hcal, HcalBarrel);
    reco::TrackBase::TrackQuality trackQuality_ = reco::TrackBase::qualityByName(theTrackQuality);

    unsigned int indx;
    reco::TrackCollection::const_iterator trkItr;
    for (trkItr = trkCollection->begin(), indx = 0; trkItr != trkCollection->end(); ++trkItr, indx++) {
      const reco::Track* pTrack = &(*trkItr);
      spr::propagatedTrackID vdet;
      vdet.trkItr = trkItr;
      vdet.ok = (trackQuality_ != reco::TrackBase::undefQuality) ? (pTrack->quality(trackQuality_)) : true;
      vdet.detIdECAL = DetId(0);
      vdet.detIdHCAL = DetId(0);
      vdet.detIdEHCAL = DetId(0);
#ifdef EDM_ML_DEBUG
      if (debug)
        std::cout << "Propagate track " << indx << " p " << trkItr->p() << " eta " << trkItr->eta() << " phi "
                  << trkItr->phi() << " Flag " << vdet.ok << std::endl;
#endif
      std::pair<math::XYZPoint, bool> info = spr::propagateECAL(pTrack, bField, debug);
      vdet.okECAL = info.second;
      if (vdet.okECAL) {
        const GlobalPoint point(info.first.x(), info.first.y(), info.first.z());
        vdet.etaECAL = point.eta();
        vdet.phiECAL = point.phi();
        if (std::abs(point.eta()) < spr::etaBEEcal) {
          vdet.detIdECAL = barrelGeom->getClosestCell(point);
        } else {
          if (endcapGeom)
            vdet.detIdECAL = endcapGeom->getClosestCell(point);
          else
            vdet.okECAL = false;
        }
        vdet.detIdEHCAL = gHB->getClosestCell(point);
      }
      info = spr::propagateHCAL(pTrack, bField, debug);
      vdet.okHCAL = info.second;
      if (vdet.okHCAL) {
        const GlobalPoint point(info.first.x(), info.first.y(), info.first.z());
        vdet.etaHCAL = point.eta();
        vdet.phiHCAL = point.phi();
        vdet.detIdHCAL = gHB->getClosestCell(point);
      }
#ifdef EDM_ML_DEBUG
      if (debug) {
        std::cout << "Track [" << indx << "] Flag: " << vdet.ok << " ECAL (" << vdet.okECAL << ") ";
        if (vdet.detIdECAL.subdetId() == EcalBarrel)
          std::cout << (EBDetId)(vdet.detIdECAL);
        else
          std::cout << (EEDetId)(vdet.detIdECAL);
        std::cout << " HCAL (" << vdet.okHCAL << ") " << (HcalDetId)(vdet.detIdHCAL) << " Or "
                  << (HcalDetId)(vdet.detIdEHCAL) << std::endl;
      }
#endif
      vdets.push_back(vdet);
    }
#ifdef EDM_ML_DEBUG
    if (debug) {
      std::cout << "propagateCALO:: for " << vdets.size() << " tracks" << std::endl;
      for (unsigned int i = 0; i < vdets.size(); ++i) {
        std::cout << "Track [" << i << "] Flag: " << vdets[i].ok << " ECAL (" << vdets[i].okECAL << ") ";
        if (vdets[i].detIdECAL.subdetId() == EcalBarrel) {
          std::cout << (EBDetId)(vdets[i].detIdECAL);
        } else {
          std::cout << (EEDetId)(vdets[i].detIdECAL);
        }
        std::cout << " HCAL (" << vdets[i].okHCAL << ") " << (HcalDetId)(vdets[i].detIdHCAL) << " Or "
                  << (HcalDetId)(vdets[i].detIdEHCAL) << std::endl;
      }
    }
#endif
  }

  void propagateCALO(edm::Handle<reco::TrackCollection>& trkCollection,
                     const CaloGeometry* geo,
                     const MagneticField* bField,
                     const std::string& theTrackQuality,
                     std::vector<spr::propagatedTrackDirection>& trkDir,
                     bool debug) {
    const CaloSubdetectorGeometry* barrelGeom = geo->getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
    const CaloSubdetectorGeometry* endcapGeom = geo->getSubdetectorGeometry(DetId::Ecal, EcalEndcap);
    const CaloSubdetectorGeometry* gHB = geo->getSubdetectorGeometry(DetId::Hcal, HcalBarrel);
    reco::TrackBase::TrackQuality trackQuality_ = reco::TrackBase::qualityByName(theTrackQuality);

    unsigned int indx;
    reco::TrackCollection::const_iterator trkItr;
    for (trkItr = trkCollection->begin(), indx = 0; trkItr != trkCollection->end(); ++trkItr, indx++) {
      const reco::Track* pTrack = &(*trkItr);
      spr::propagatedTrackDirection trkD;
      trkD.trkItr = trkItr;
      trkD.ok = (trackQuality_ != reco::TrackBase::undefQuality) ? (pTrack->quality(trackQuality_)) : true;
      trkD.detIdECAL = DetId(0);
      trkD.detIdHCAL = DetId(0);
      trkD.detIdEHCAL = DetId(0);
#ifdef EDM_ML_DEBUG
      if (debug)
        std::cout << "Propagate track " << indx << " p " << trkItr->p() << " eta " << trkItr->eta() << " phi "
                  << trkItr->phi() << " Flag " << trkD.ok << std::endl;
#endif
      spr::propagatedTrack info = spr::propagateTrackToECAL(pTrack, bField, debug);
      GlobalPoint point(info.point.x(), info.point.y(), info.point.z());
      trkD.okECAL = info.ok;
      trkD.pointECAL = point;
      trkD.directionECAL = info.direction;
      if (trkD.okECAL) {
        if (std::abs(info.point.eta()) < spr::etaBEEcal) {
          trkD.detIdECAL = barrelGeom->getClosestCell(point);
        } else {
          if (endcapGeom)
            trkD.detIdECAL = endcapGeom->getClosestCell(point);
          else
            trkD.okECAL = false;
        }
        trkD.detIdEHCAL = gHB->getClosestCell(point);
      }
      info = spr::propagateTrackToHCAL(pTrack, bField, debug);
      point = GlobalPoint(info.point.x(), info.point.y(), info.point.z());
      trkD.okHCAL = info.ok;
      trkD.pointHCAL = point;
      trkD.directionHCAL = info.direction;
      if (trkD.okHCAL) {
        trkD.detIdHCAL = gHB->getClosestCell(point);
      }
      trkDir.push_back(trkD);
    }
#ifdef EDM_ML_DEBUG
    if (debug) {
      std::cout << "propagateCALO:: for " << trkDir.size() << " tracks" << std::endl;
      for (unsigned int i = 0; i < trkDir.size(); ++i) {
        std::cout << "Track [" << i << "] Flag: " << trkDir[i].ok << " ECAL (" << trkDir[i].okECAL << ")";
        if (trkDir[i].okECAL) {
          std::cout << " point " << trkDir[i].pointECAL << " direction " << trkDir[i].directionECAL << " ";
          if (trkDir[i].detIdECAL.subdetId() == EcalBarrel) {
            std::cout << (EBDetId)(trkDir[i].detIdECAL);
          } else {
            std::cout << (EEDetId)(trkDir[i].detIdECAL);
          }
        }
        std::cout << " HCAL (" << trkDir[i].okHCAL << ")";
        if (trkDir[i].okHCAL) {
          std::cout << " point " << trkDir[i].pointHCAL << " direction " << trkDir[i].directionHCAL << " "
                    << (HcalDetId)(trkDir[i].detIdHCAL);
        }
        std::cout << " Or " << (HcalDetId)(trkDir[i].detIdEHCAL) << std::endl;
      }
    }
#endif
  }

  spr::propagatedTrackID propagateCALO(const reco::Track* pTrack,
                                       const CaloGeometry* geo,
                                       const MagneticField* bField,
                                       bool debug) {
    const CaloSubdetectorGeometry* barrelGeom = geo->getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
    const CaloSubdetectorGeometry* endcapGeom = geo->getSubdetectorGeometry(DetId::Ecal, EcalEndcap);
    const CaloSubdetectorGeometry* gHB = geo->getSubdetectorGeometry(DetId::Hcal, HcalBarrel);

    spr::propagatedTrackID vdet;
    vdet.ok = true;
    vdet.detIdECAL = DetId(0);
    vdet.detIdHCAL = DetId(0);
    vdet.detIdEHCAL = DetId(0);
#ifdef EDM_ML_DEBUG
    if (debug)
      std::cout << "Propagate track:  p " << pTrack->p() << " eta " << pTrack->eta() << " phi " << pTrack->phi()
                << " Flag " << vdet.ok << std::endl;
#endif
    std::pair<math::XYZPoint, bool> info = spr::propagateECAL(pTrack, bField, debug);
    vdet.okECAL = info.second;
    if (vdet.okECAL) {
      const GlobalPoint point(info.first.x(), info.first.y(), info.first.z());
      vdet.etaECAL = point.eta();
      vdet.phiECAL = point.phi();
      if (std::abs(point.eta()) < spr::etaBEEcal) {
        vdet.detIdECAL = barrelGeom->getClosestCell(point);
      } else {
        if (endcapGeom)
          vdet.detIdECAL = endcapGeom->getClosestCell(point);
        else
          vdet.okECAL = false;
      }
      vdet.detIdEHCAL = gHB->getClosestCell(point);
    }
    info = spr::propagateHCAL(pTrack, bField, debug);
    vdet.okHCAL = info.second;
    if (vdet.okHCAL) {
      const GlobalPoint point(info.first.x(), info.first.y(), info.first.z());
      vdet.etaHCAL = point.eta();
      vdet.phiHCAL = point.phi();
      vdet.detIdHCAL = gHB->getClosestCell(point);
    }
#ifdef EDM_ML_DEBUG
    if (debug) {
      std::cout << "propagateCALO:: for 1 track" << std::endl;
      std::cout << "Track [0] Flag: " << vdet.ok << " ECAL (" << vdet.okECAL << ") ";
      if (vdet.detIdECAL.subdetId() == EcalBarrel) {
        std::cout << (EBDetId)(vdet.detIdECAL);
      } else {
        std::cout << (EEDetId)(vdet.detIdECAL);
      }
      std::cout << " HCAL (" << vdet.okHCAL << ") " << (HcalDetId)(vdet.detIdHCAL) << " Or "
                << (HcalDetId)(vdet.detIdEHCAL) << std::endl;
    }
#endif
    return vdet;
  }

  std::vector<spr::propagatedGenTrackID> propagateCALO(const HepMC::GenEvent* genEvent,
                                                       const ParticleDataTable* pdt,
                                                       const CaloGeometry* geo,
                                                       const MagneticField* bField,
                                                       double etaMax,
                                                       bool debug) {
    const CaloSubdetectorGeometry* barrelGeom = geo->getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
    const CaloSubdetectorGeometry* endcapGeom = geo->getSubdetectorGeometry(DetId::Ecal, EcalEndcap);
    const CaloSubdetectorGeometry* gHB = geo->getSubdetectorGeometry(DetId::Hcal, HcalBarrel);

    std::vector<spr::propagatedGenTrackID> trkDir;
    unsigned int indx;
    HepMC::GenEvent::particle_const_iterator p;
    for (p = genEvent->particles_begin(), indx = 0; p != genEvent->particles_end(); ++p, ++indx) {
      spr::propagatedGenTrackID trkD;
      trkD.trkItr = p;
      trkD.detIdECAL = DetId(0);
      trkD.detIdHCAL = DetId(0);
      trkD.detIdEHCAL = DetId(0);
      trkD.pdgId = ((*p)->pdg_id());
      trkD.charge = ((pdt->particle(trkD.pdgId))->ID().threeCharge()) / 3;
      const GlobalVector momentum = GlobalVector((*p)->momentum().px(), (*p)->momentum().py(), (*p)->momentum().pz());
#ifdef EDM_ML_DEBUG
      if (debug)
        std::cout << "Propagate track " << indx << " pdg " << trkD.pdgId << " charge " << trkD.charge << " p "
                  << momentum << std::endl;
#endif
      // consider stable particles
      if ((*p)->status() == 1 && std::abs((*p)->momentum().eta()) < etaMax) {
        const GlobalPoint vertex = GlobalPoint(0.1 * (*p)->production_vertex()->position().x(),
                                               0.1 * (*p)->production_vertex()->position().y(),
                                               0.1 * (*p)->production_vertex()->position().z());
        trkD.ok = true;
        spr::propagatedTrack info = spr::propagateCalo(
            vertex, momentum, trkD.charge, bField, spr::zFrontEE, spr::rFrontEB, spr::etaBEEcal, debug);
        GlobalPoint point(info.point.x(), info.point.y(), info.point.z());
        trkD.okECAL = info.ok;
        trkD.pointECAL = point;
        trkD.directionECAL = info.direction;
        if (trkD.okECAL) {
          if (std::abs(info.point.eta()) < spr::etaBEEcal) {
            trkD.detIdECAL = barrelGeom->getClosestCell(point);
          } else {
            if (endcapGeom)
              trkD.detIdECAL = endcapGeom->getClosestCell(point);
            else
              trkD.okECAL = false;
          }
          trkD.detIdEHCAL = gHB->getClosestCell(point);
        }

        info = spr::propagateCalo(
            vertex, momentum, trkD.charge, bField, spr::zFrontHE, spr::rFrontHB, spr::etaBEHcal, debug);
        point = GlobalPoint(info.point.x(), info.point.y(), info.point.z());
        trkD.okHCAL = info.ok;
        trkD.pointHCAL = point;
        trkD.directionHCAL = info.direction;
        if (trkD.okHCAL) {
          trkD.detIdHCAL = gHB->getClosestCell(point);
        }
      }
      trkDir.push_back(trkD);
    }
#ifdef EDM_ML_DEBUG
    if (debug) {
      std::cout << "propagateCALO:: for " << trkDir.size() << " tracks" << std::endl;
      for (unsigned int i = 0; i < trkDir.size(); ++i) {
        if (trkDir[i].okECAL)
          std::cout << "Track [" << i << "] Flag: " << trkDir[i].ok << " ECAL (" << trkDir[i].okECAL << ")";
        if (trkDir[i].okECAL) {
          std::cout << " point " << trkDir[i].pointECAL << " direction " << trkDir[i].directionECAL << " ";
          if (trkDir[i].detIdECAL.subdetId() == EcalBarrel) {
            std::cout << (EBDetId)(trkDir[i].detIdECAL);
          } else {
            std::cout << (EEDetId)(trkDir[i].detIdECAL);
          }
        }
        if (trkDir[i].okECAL)
          std::cout << " HCAL (" << trkDir[i].okHCAL << ")";
        if (trkDir[i].okHCAL) {
          std::cout << " point " << trkDir[i].pointHCAL << " direction " << trkDir[i].directionHCAL << " "
                    << (HcalDetId)(trkDir[i].detIdHCAL);
        }
        if (trkDir[i].okECAL)
          std::cout << " Or " << (HcalDetId)(trkDir[i].detIdEHCAL) << std::endl;
      }
    }
#endif
    return trkDir;
  }

  std::vector<spr::propagatedGenParticleID> propagateCALO(edm::Handle<reco::GenParticleCollection>& genParticles,
                                                          const ParticleDataTable* pdt,
                                                          const CaloGeometry* geo,
                                                          const MagneticField* bField,
                                                          double etaMax,
                                                          bool debug) {
    const CaloSubdetectorGeometry* barrelGeom = geo->getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
    const CaloSubdetectorGeometry* endcapGeom = geo->getSubdetectorGeometry(DetId::Ecal, EcalEndcap);
    const CaloSubdetectorGeometry* gHB = geo->getSubdetectorGeometry(DetId::Hcal, HcalBarrel);

    std::vector<spr::propagatedGenParticleID> trkDir;
    unsigned int indx;
    reco::GenParticleCollection::const_iterator p;
    for (p = genParticles->begin(), indx = 0; p != genParticles->end(); ++p, ++indx) {
      spr::propagatedGenParticleID trkD;
      trkD.trkItr = p;
      trkD.detIdECAL = DetId(0);
      trkD.detIdHCAL = DetId(0);
      trkD.detIdEHCAL = DetId(0);
      trkD.pdgId = (p->pdgId());
      trkD.charge = p->charge();
      const GlobalVector momentum = GlobalVector(p->momentum().x(), p->momentum().y(), p->momentum().z());
#ifdef EDM_ML_DEBUG
      if (debug)
        std::cout << "Propagate track " << indx << " pdg " << trkD.pdgId << " charge " << trkD.charge << " p "
                  << momentum << std::endl;
#endif
      // consider stable particles
      if (p->status() == 1 && std::abs(momentum.eta()) < etaMax) {
        const GlobalPoint vertex = GlobalPoint(p->vertex().x(), p->vertex().y(), p->vertex().z());
        trkD.ok = true;
        spr::propagatedTrack info = spr::propagateCalo(
            vertex, momentum, trkD.charge, bField, spr::zFrontEE, spr::rFrontEB, spr::etaBEEcal, debug);
        GlobalPoint point(info.point.x(), info.point.y(), info.point.z());
        trkD.okECAL = info.ok;
        trkD.pointECAL = point;
        trkD.directionECAL = info.direction;
        if (trkD.okECAL) {
          if (std::abs(info.point.eta()) < spr::etaBEEcal) {
            trkD.detIdECAL = barrelGeom->getClosestCell(point);
          } else {
            if (endcapGeom)
              trkD.detIdECAL = endcapGeom->getClosestCell(point);
            else
              trkD.okECAL = false;
          }
          trkD.detIdEHCAL = gHB->getClosestCell(point);
        }

        info = spr::propagateCalo(
            vertex, momentum, trkD.charge, bField, spr::zFrontHE, spr::rFrontHB, spr::etaBEHcal, debug);
        point = GlobalPoint(info.point.x(), info.point.y(), info.point.z());
        trkD.okHCAL = info.ok;
        trkD.pointHCAL = point;
        trkD.directionHCAL = info.direction;
        if (trkD.okHCAL) {
          trkD.detIdHCAL = gHB->getClosestCell(point);
        }
      }
      trkDir.push_back(trkD);
    }
#ifdef EDM_ML_DEBUG
    if (debug) {
      std::cout << "propagateCALO:: for " << trkDir.size() << " tracks" << std::endl;
      for (unsigned int i = 0; i < trkDir.size(); ++i) {
        if (trkDir[i].okECAL)
          std::cout << "Track [" << i << "] Flag: " << trkDir[i].ok << " ECAL (" << trkDir[i].okECAL << ")";
        if (trkDir[i].okECAL) {
          std::cout << " point " << trkDir[i].pointECAL << " direction " << trkDir[i].directionECAL << " ";
          if (trkDir[i].detIdECAL.subdetId() == EcalBarrel) {
            std::cout << (EBDetId)(trkDir[i].detIdECAL);
          } else {
            std::cout << (EEDetId)(trkDir[i].detIdECAL);
          }
        }
        if (trkDir[i].okECAL)
          std::cout << " HCAL (" << trkDir[i].okHCAL << ")";
        if (trkDir[i].okHCAL) {
          std::cout << " point " << trkDir[i].pointHCAL << " direction " << trkDir[i].directionHCAL << " "
                    << (HcalDetId)(trkDir[i].detIdHCAL);
        }
        if (trkDir[i].okECAL)
          std::cout << " Or " << (HcalDetId)(trkDir[i].detIdEHCAL) << std::endl;
      }
    }
#endif
    return trkDir;
  }

  spr::propagatedTrackDirection propagateCALO(unsigned int thisTrk,
                                              edm::Handle<edm::SimTrackContainer>& SimTk,
                                              edm::Handle<edm::SimVertexContainer>& SimVtx,
                                              const CaloGeometry* geo,
                                              const MagneticField* bField,
                                              bool debug) {
    const CaloSubdetectorGeometry* barrelGeom = geo->getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
    const CaloSubdetectorGeometry* endcapGeom = geo->getSubdetectorGeometry(DetId::Ecal, EcalEndcap);
    const CaloSubdetectorGeometry* gHB = geo->getSubdetectorGeometry(DetId::Hcal, HcalBarrel);

    spr::trackAtOrigin trk = spr::simTrackAtOrigin(thisTrk, SimTk, SimVtx, debug);
    spr::propagatedTrackDirection trkD;
    trkD.ok = trk.ok;
    trkD.detIdECAL = DetId(0);
    trkD.detIdHCAL = DetId(0);
    trkD.detIdEHCAL = DetId(0);
#ifdef EDM_ML_DEBUG
    if (debug)
      std::cout << "Propagate track " << thisTrk << " charge " << trk.charge << " position " << trk.position << " p "
                << trk.momentum << " Flag " << trkD.ok << std::endl;
#endif
    if (trkD.ok) {
      spr::propagatedTrack info = spr::propagateCalo(
          trk.position, trk.momentum, trk.charge, bField, spr::zFrontEE, spr::rFrontEB, spr::etaBEEcal, debug);
      GlobalPoint point(info.point.x(), info.point.y(), info.point.z());
      trkD.okECAL = info.ok;
      trkD.pointECAL = point;
      trkD.directionECAL = info.direction;
      if (trkD.okECAL) {
        if (std::abs(info.point.eta()) < spr::etaBEEcal) {
          trkD.detIdECAL = barrelGeom->getClosestCell(point);
        } else {
          if (endcapGeom)
            trkD.detIdECAL = endcapGeom->getClosestCell(point);
          else
            trkD.okECAL = false;
        }
        trkD.detIdEHCAL = gHB->getClosestCell(point);
      }

      info = spr::propagateCalo(
          trk.position, trk.momentum, trk.charge, bField, spr::zFrontHE, spr::rFrontHB, spr::etaBEHcal, debug);
      point = GlobalPoint(info.point.x(), info.point.y(), info.point.z());
      trkD.okHCAL = info.ok;
      trkD.pointHCAL = point;
      trkD.directionHCAL = info.direction;
      if (trkD.okHCAL) {
        trkD.detIdHCAL = gHB->getClosestCell(point);
      }
    }
#ifdef EDM_ML_DEBUG
    if (debug) {
      std::cout << "propagateCALO:: for track [" << thisTrk << "] Flag: " << trkD.ok << " ECAL (" << trkD.okECAL
                << ") HCAL (" << trkD.okHCAL << ")" << std::endl;
      if (trkD.okECAL) {
        std::cout << "ECAL point " << trkD.pointECAL << " direction " << trkD.directionECAL << " ";
        if (trkD.detIdECAL.subdetId() == EcalBarrel) {
          std::cout << (EBDetId)(trkD.detIdECAL);
        } else {
          std::cout << (EEDetId)(trkD.detIdECAL);
        }
      }
      if (trkD.okHCAL) {
        std::cout << " HCAL point " << trkD.pointHCAL << " direction " << trkD.directionHCAL << " "
                  << (HcalDetId)(trkD.detIdHCAL);
      }
      if (trkD.okECAL)
        std::cout << " Or " << (HcalDetId)(trkD.detIdEHCAL);
      std::cout << std::endl;
    }
#endif
    return trkD;
  }

  spr::propagatedTrackDirection propagateHCALBack(unsigned int thisTrk,
                                                  edm::Handle<edm::SimTrackContainer>& SimTk,
                                                  edm::Handle<edm::SimVertexContainer>& SimVtx,
                                                  const CaloGeometry* geo,
                                                  const MagneticField* bField,
                                                  bool debug) {
    const CaloSubdetectorGeometry* gHB = geo->getSubdetectorGeometry(DetId::Hcal, HcalBarrel);
    spr::trackAtOrigin trk = spr::simTrackAtOrigin(thisTrk, SimTk, SimVtx, debug);
    spr::propagatedTrackDirection trkD;
    trkD.ok = trk.ok;
    trkD.detIdECAL = DetId(0);
    trkD.detIdHCAL = DetId(0);
    trkD.detIdEHCAL = DetId(0);
#ifdef EDM_ML_DEBUG
    if (debug)
      std::cout << "Propagate track " << thisTrk << " charge " << trk.charge << " position " << trk.position << " p "
                << trk.momentum << " Flag " << trkD.ok << std::endl;
#endif
    if (trkD.ok) {
      spr::propagatedTrack info = spr::propagateCalo(
          trk.position, trk.momentum, trk.charge, bField, spr::zBackHE, spr::rBackHB, spr::etaBEHcal, debug);
      const GlobalPoint point = GlobalPoint(info.point.x(), info.point.y(), info.point.z());
      trkD.okHCAL = info.ok;
      trkD.pointHCAL = point;
      trkD.directionHCAL = info.direction;
      if (trkD.okHCAL) {
        trkD.detIdHCAL = gHB->getClosestCell(point);
      }
    }
#ifdef EDM_ML_DEBUG
    if (debug) {
      std::cout << "propagateCALO:: for track [" << thisTrk << "] Flag: " << trkD.ok << " ECAL (" << trkD.okECAL
                << ") HCAL (" << trkD.okHCAL << ")" << std::endl;
      if (trkD.okHCAL) {
        std::cout << " HCAL point " << trkD.pointHCAL << " direction " << trkD.directionHCAL << " "
                  << (HcalDetId)(trkD.detIdHCAL);
      }
    }
#endif
    return trkD;
  }

  std::pair<bool, HcalDetId> propagateHCALBack(const reco::Track* track,
                                               const CaloGeometry* geo,
                                               const MagneticField* bField,
                                               bool debug) {
    const CaloSubdetectorGeometry* gHB = geo->getSubdetectorGeometry(DetId::Hcal, HcalBarrel);
    const GlobalPoint vertex(track->vx(), track->vy(), track->vz());
    const GlobalVector momentum(track->px(), track->py(), track->pz());
    int charge(track->charge());
    spr::propagatedTrack info =
        spr::propagateCalo(vertex, momentum, charge, bField, spr::zBackHE, spr::rBackHB, spr::etaBEHcal, debug);
    if (info.ok) {
      const GlobalPoint point = GlobalPoint(info.point.x(), info.point.y(), info.point.z());
      return std::pair<bool, HcalDetId>(true, HcalDetId(gHB->getClosestCell(point)));
    } else {
      return std::pair<bool, HcalDetId>(false, HcalDetId());
    }
  }

  propagatedTrack propagateTrackToECAL(const reco::Track* track, const MagneticField* bfield, bool debug) {
    const GlobalPoint vertex(track->vx(), track->vy(), track->vz());
    const GlobalVector momentum(track->px(), track->py(), track->pz());
    int charge(track->charge());
    return spr::propagateCalo(vertex, momentum, charge, bfield, spr::zFrontEE, spr::rFrontEB, spr::etaBEEcal, debug);
  }

  propagatedTrack propagateTrackToECAL(unsigned int thisTrk,
                                       edm::Handle<edm::SimTrackContainer>& SimTk,
                                       edm::Handle<edm::SimVertexContainer>& SimVtx,
                                       const MagneticField* bfield,
                                       bool debug) {
    spr::trackAtOrigin trk = spr::simTrackAtOrigin(thisTrk, SimTk, SimVtx, debug);
    spr::propagatedTrack ptrk;
    if (trk.ok)
      ptrk = spr::propagateCalo(
          trk.position, trk.momentum, trk.charge, bfield, spr::zFrontEE, spr::rFrontEB, spr::etaBEEcal, debug);
    return ptrk;
  }

  std::pair<math::XYZPoint, bool> propagateECAL(const reco::Track* track, const MagneticField* bfield, bool debug) {
    const GlobalPoint vertex(track->vx(), track->vy(), track->vz());
    const GlobalVector momentum(track->px(), track->py(), track->pz());
    int charge(track->charge());
    return spr::propagateECAL(vertex, momentum, charge, bfield, debug);
  }

  std::pair<DetId, bool> propagateIdECAL(const HcalDetId& id,
                                         const CaloGeometry* geo,
                                         const MagneticField* bField,
                                         bool debug) {
    const HcalGeometry* gHB = static_cast<const HcalGeometry*>(geo->getSubdetectorGeometry(DetId::Hcal, HcalBarrel));
    const GlobalPoint vertex(0, 0, 0);
    const GlobalPoint hit(gHB->getPosition(id));
    const GlobalVector momentum = GlobalVector(hit.x(), hit.y(), hit.z());
    std::pair<math::XYZPoint, bool> info = propagateECAL(vertex, momentum, 0, bField, debug);
    DetId eId(0);
    if (info.second) {
      const GlobalPoint point(info.first.x(), info.first.y(), info.first.z());
      if (std::abs(point.eta()) < spr::etaBEEcal) {
        const CaloSubdetectorGeometry* barrelGeom = geo->getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
        eId = barrelGeom->getClosestCell(point);
      } else {
        const CaloSubdetectorGeometry* endcapGeom = geo->getSubdetectorGeometry(DetId::Ecal, EcalEndcap);
        if (endcapGeom)
          eId = endcapGeom->getClosestCell(point);
        else
          info.second = false;
      }
    }
    return std::pair<DetId, bool>(eId, info.second);
  }

  std::pair<math::XYZPoint, bool> propagateECAL(
      const GlobalPoint& vertex, const GlobalVector& momentum, int charge, const MagneticField* bfield, bool debug) {
    spr::propagatedTrack track =
        spr::propagateCalo(vertex, momentum, charge, bfield, spr::zFrontEE, spr::rFrontEB, spr::etaBEEcal, debug);
    return std::pair<math::XYZPoint, bool>(track.point, track.ok);
  }

  spr::propagatedTrack propagateTrackToHCAL(const reco::Track* track, const MagneticField* bfield, bool debug) {
    const GlobalPoint vertex(track->vx(), track->vy(), track->vz());
    const GlobalVector momentum(track->px(), track->py(), track->pz());
    int charge(track->charge());
    return spr::propagateCalo(vertex, momentum, charge, bfield, spr::zFrontHE, spr::rFrontHB, spr::etaBEHcal, debug);
  }

  spr::propagatedTrack propagateTrackToHCAL(unsigned int thisTrk,
                                            edm::Handle<edm::SimTrackContainer>& SimTk,
                                            edm::Handle<edm::SimVertexContainer>& SimVtx,
                                            const MagneticField* bfield,
                                            bool debug) {
    spr::trackAtOrigin trk = spr::simTrackAtOrigin(thisTrk, SimTk, SimVtx, debug);
    spr::propagatedTrack ptrk;
    if (trk.ok)
      ptrk = spr::propagateCalo(
          trk.position, trk.momentum, trk.charge, bfield, spr::zFrontHE, spr::rFrontHB, spr::etaBEHcal, debug);
    return ptrk;
  }

  std::pair<math::XYZPoint, bool> propagateHCAL(const reco::Track* track, const MagneticField* bfield, bool debug) {
    const GlobalPoint vertex(track->vx(), track->vy(), track->vz());
    const GlobalVector momentum(track->px(), track->py(), track->pz());
    int charge(track->charge());
    return spr::propagateHCAL(vertex, momentum, charge, bfield, debug);
  }

  std::pair<math::XYZPoint, bool> propagateHCAL(
      const GlobalPoint& vertex, const GlobalVector& momentum, int charge, const MagneticField* bfield, bool debug) {
    spr::propagatedTrack track =
        spr::propagateCalo(vertex, momentum, charge, bfield, spr::zFrontHE, spr::rFrontHB, spr::etaBEHcal, debug);
    return std::pair<math::XYZPoint, bool>(track.point, track.ok);
  }

  std::pair<math::XYZPoint, bool> propagateTracker(const reco::Track* track, const MagneticField* bfield, bool debug) {
    const GlobalPoint vertex(track->vx(), track->vy(), track->vz());
    const GlobalVector momentum(track->px(), track->py(), track->pz());
    int charge(track->charge());
    spr::propagatedTrack track1 =
        spr::propagateCalo(vertex, momentum, charge, bfield, spr::zBackTE, spr::rBackTB, spr::etaBETrak, debug);
    return std::pair<math::XYZPoint, bool>(track1.point, track1.ok);
  }

  std::pair<math::XYZPoint, double> propagateTrackerEnd(const reco::Track* track,
                                                        const MagneticField* bField,
                                                        bool
#ifdef EDM_ML_DEBUG
                                                            debug
#endif
  ) {

    const GlobalPoint vertex(track->vx(), track->vy(), track->vz());
    const GlobalVector momentum(track->px(), track->py(), track->pz());
    int charge(track->charge());
    float radius = track->outerPosition().Rho();
    float zdist = track->outerPosition().Z();
#ifdef EDM_ML_DEBUG
    if (debug)
      std::cout << "propagateTrackerEnd:: Vertex " << vertex << " Momentum " << momentum << " Charge " << charge
                << " Radius " << radius << " Z " << zdist << std::endl;
#endif
    FreeTrajectoryState fts(vertex, momentum, charge, bField);
    Plane::PlanePointer endcap = Plane::build(Plane::PositionType(0, 0, zdist), Plane::RotationType());
    Cylinder::CylinderPointer barrel =
        Cylinder::build(Cylinder::PositionType(0, 0, 0), Cylinder::RotationType(), radius);

    AnalyticalPropagator myAP(bField, alongMomentum, 2 * M_PI);

    TrajectoryStateOnSurface tsose = myAP.propagate(fts, *endcap);
    TrajectoryStateOnSurface tsosb = myAP.propagate(fts, *barrel);

    math::XYZPoint point(-999., -999., -999.);
    bool ok = false;
    GlobalVector direction(0, 0, 1);
    if (tsosb.isValid() && std::abs(zdist) < spr::zFrontTE) {
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
      math::XYZPoint vDiff(point.x() - vertex.x(), point.y() - vertex.y(), point.z() - vertex.z());
      double dphi = direction.phi() - momentum.phi();
      double rdist = std::sqrt(vDiff.x() * vDiff.x() + vDiff.y() * vDiff.y());
      double rat = 0.5 * dphi / std::sin(0.5 * dphi);
      double dZ = vDiff.z();
      double dS = rdist * rat;  //dZ*momentum.z()/momentum.perp();
      length = std::sqrt(dS * dS + dZ * dZ);
#ifdef EDM_ML_DEBUG
      if (debug)
        std::cout << "propagateTracker:: Barrel " << tsosb.isValid() << " Endcap " << tsose.isValid() << " OverAll "
                  << ok << " Point " << point << " RDist " << rdist << " dS " << dS << " dS/pt "
                  << rdist * rat / momentum.perp() << " zdist " << dZ << " dz/pz " << dZ / momentum.z() << " Length "
                  << length << std::endl;
#endif
    }

    return std::pair<math::XYZPoint, double>(point, length);
  }

  spr::propagatedTrack propagateCalo(const GlobalPoint& tpVertex,
                                     const GlobalVector& tpMomentum,
                                     int tpCharge,
                                     const MagneticField* bField,
                                     float zdist,
                                     float radius,
                                     float corner,
                                     bool
#ifdef EDM_ML_DEBUG
                                         debug
#endif
  ) {

    spr::propagatedTrack track;
#ifdef EDM_ML_DEBUG
    if (debug)
      std::cout << "propagateCalo:: Vertex " << tpVertex << " Momentum " << tpMomentum << " Charge " << tpCharge
                << " Radius " << radius << " Z " << zdist << " Corner " << corner << std::endl;
#endif
    FreeTrajectoryState fts(tpVertex, tpMomentum, tpCharge, bField);

    Plane::PlanePointer lendcap = Plane::build(Plane::PositionType(0, 0, -zdist), Plane::RotationType());
    Plane::PlanePointer rendcap = Plane::build(Plane::PositionType(0, 0, zdist), Plane::RotationType());

    Cylinder::CylinderPointer barrel =
        Cylinder::build(Cylinder::PositionType(0, 0, 0), Cylinder::RotationType(), radius);

    AnalyticalPropagator myAP(bField, alongMomentum, 2 * M_PI);

    TrajectoryStateOnSurface tsose;
    if (tpMomentum.eta() < 0) {
      tsose = myAP.propagate(fts, *lendcap);
    } else {
      tsose = myAP.propagate(fts, *rendcap);
    }

    TrajectoryStateOnSurface tsosb = myAP.propagate(fts, *barrel);

    track.ok = true;
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
      track.direction = GlobalVector(0, 0, 1);
      track.ok = false;
    }
#ifdef EDM_ML_DEBUG
    if (debug) {
      std::cout << "propagateCalo:: Barrel " << tsosb.isValid() << " Endcap " << tsose.isValid() << " OverAll "
                << track.ok << " Point " << track.point << " Direction " << track.direction << std::endl;
      if (track.ok) {
        math::XYZPoint vDiff(
            track.point.x() - tpVertex.x(), track.point.y() - tpVertex.y(), track.point.z() - tpVertex.z());
        double dphi = track.direction.phi() - tpMomentum.phi();
        double rdist = std::sqrt(vDiff.x() * vDiff.x() + vDiff.y() * vDiff.y());
        double pt = tpMomentum.perp();
        double rat = 0.5 * dphi / std::sin(0.5 * dphi);
        std::cout << "RDist " << rdist << " pt " << pt << " r/pt " << rdist * rat / pt << " zdist " << vDiff.z()
                  << " pz " << tpMomentum.z() << " z/pz " << vDiff.z() / tpMomentum.z() << std::endl;
      }
    }
#endif
    return track;
  }

  spr::trackAtOrigin simTrackAtOrigin(unsigned int thisTrk,
                                      edm::Handle<edm::SimTrackContainer>& SimTk,
                                      edm::Handle<edm::SimVertexContainer>& SimVtx,
                                      bool
#ifdef EDM_ML_DEBUG
                                          debug
#endif
  ) {

    spr::trackAtOrigin trk;

    edm::SimTrackContainer::const_iterator itr = SimTk->end();
    for (edm::SimTrackContainer::const_iterator simTrkItr = SimTk->begin(); simTrkItr != SimTk->end(); simTrkItr++) {
      if (simTrkItr->trackId() == thisTrk) {
#ifdef EDM_ML_DEBUG
        if (debug)
          std::cout << "matched trackId (maximum occurance) " << thisTrk << " type " << simTrkItr->type() << std::endl;
#endif
        itr = simTrkItr;
        break;
      }
    }

    if (itr != SimTk->end()) {
      int vertIndex = itr->vertIndex();
      if (vertIndex != -1 && vertIndex < (int)SimVtx->size()) {
        edm::SimVertexContainer::const_iterator simVtxItr = SimVtx->begin();
        for (int iv = 0; iv < vertIndex; iv++)
          simVtxItr++;
        const math::XYZTLorentzVectorD pos = simVtxItr->position();
        const math::XYZTLorentzVectorD mom = itr->momentum();
        trk.ok = true;
        trk.charge = (int)(itr->charge());
        trk.position = GlobalPoint(pos.x(), pos.y(), pos.z());
        trk.momentum = GlobalVector(mom.x(), mom.y(), mom.z());
      }
    }
#ifdef EDM_ML_DEBUG
    if (debug)
      std::cout << "Track flag " << trk.ok << " Position " << trk.position << " Momentum " << trk.momentum << std::endl;
#endif
    return trk;
  }

  bool propagateHCAL(const reco::Track* track,
                     const CaloGeometry* geo,
                     const MagneticField* bField,
                     bool typeRZ,
                     const std::pair<double, double> rz,
                     bool debug) {
    const GlobalPoint vertex(track->vx(), track->vy(), track->vz());
    const GlobalVector momentum(track->px(), track->py(), track->pz());
    int charge(track->charge());
#ifdef EDM_ML_DEBUG
    if (debug)
      std::cout << "Propagate track with charge " << charge << " position " << vertex << " p " << momentum << std::endl;
#endif
    std::pair<HcalDetId, HcalDetId> ids = propagateHCAL(geo, bField, vertex, momentum, charge, typeRZ, rz, debug);
    bool ok = ((ids.first != HcalDetId()) && (ids.first.ieta() == ids.second.ieta()) &&
               (ids.first.iphi() == ids.second.iphi()));
    return ok;
  }

  bool propagateHCAL(unsigned int thisTrk,
                     edm::Handle<edm::SimTrackContainer>& SimTk,
                     edm::Handle<edm::SimVertexContainer>& SimVtx,
                     const CaloGeometry* geo,
                     const MagneticField* bField,
                     bool typeRZ,
                     const std::pair<double, double> rz,
                     bool debug) {
    spr::trackAtOrigin trk = spr::simTrackAtOrigin(thisTrk, SimTk, SimVtx, debug);
#ifdef EDM_ML_DEBUG
    if (debug)
      std::cout << "Propagate track " << thisTrk << " charge " << trk.charge << " position " << trk.position << " p "
                << trk.momentum << std::endl;
#endif
    std::pair<HcalDetId, HcalDetId> ids =
        propagateHCAL(geo, bField, trk.position, trk.momentum, trk.charge, typeRZ, rz, debug);
    bool ok = ((ids.first != HcalDetId()) && (ids.first.ieta() == ids.second.ieta()) &&
               (ids.first.iphi() == ids.second.iphi()));
    return ok;
  }

  std::pair<HcalDetId, HcalDetId> propagateHCAL(const CaloGeometry* geo,
                                                const MagneticField* bField,
                                                const GlobalPoint& vertex,
                                                const GlobalVector& momentum,
                                                int charge,
                                                bool typeRZ,
                                                const std::pair<double, double> rz,
                                                bool
#ifdef EDM_ML_DEBUG
                                                    debug
#endif
  ) {

#ifdef EDM_ML_DEBUG
    if (debug)
      std::cout << "propagateCalo:: Vertex " << vertex << " Momentum " << momentum << " Charge " << charge << " R/Z "
                << rz.first << " : " << rz.second << " Type " << typeRZ << std::endl;
#endif
    const CaloSubdetectorGeometry* gHB = geo->getSubdetectorGeometry(DetId::Hcal, HcalBarrel);
    FreeTrajectoryState fts(vertex, momentum, charge, bField);
    AnalyticalPropagator myAP(bField, alongMomentum, 2 * M_PI);

    HcalDetId id1, id2;
    for (int k = 0; k < 2; ++k) {
      TrajectoryStateOnSurface tsos;
      double rzv = (k == 0) ? rz.first : rz.second;
      if (typeRZ) {
        Cylinder::CylinderPointer barrel =
            Cylinder::build(Cylinder::PositionType(0, 0, 0), Cylinder::RotationType(), rzv);
        tsos = myAP.propagate(fts, *barrel);
      } else {
        Plane::PlanePointer endcap = Plane::build(Plane::PositionType(0, 0, rzv), Plane::RotationType());
        tsos = myAP.propagate(fts, *endcap);
      }

      if (tsos.isValid()) {
        GlobalPoint point = tsos.globalPosition();
        if (k == 0)
          id1 = gHB->getClosestCell(point);
        else
          id2 = gHB->getClosestCell(point);
#ifdef EDM_ML_DEBUG
        if (debug) {
          std::cout << "Iteration " << k << " Point " << point << " ID ";
          if (k == 0)
            std::cout << id1;
          else
            std::cout << id2;
          std::cout << std::endl;
        }
#endif
      }
    }
#ifdef EDM_ML_DEBUG
    if (debug)
      std::cout << "propagateCalo:: Front " << id1 << " Back " << id2 << std::endl;
#endif
    return std::pair<HcalDetId, HcalDetId>(id1, id2);
  }
}  // namespace spr
