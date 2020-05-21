#include "Calibration/IsolatedParticles/interface/MatrixECALDetIds.h"
#include "Calibration/IsolatedParticles/interface/MatrixHCALDetIds.h"
#include "Calibration/IsolatedParticles/interface/ChargeIsolation.h"
#include "Calibration/IsolatedParticles/interface/GenSimInfo.h"
#include "Calibration/IsolatedParticles/interface/DebugInfo.h"

#include <iostream>

//#define EDM_ML_DEBUG

namespace spr {

  void eGenSimInfo(const DetId& coreDet,
                   HepMC::GenEvent::particle_const_iterator trkItr,
                   std::vector<spr::propagatedGenTrackID>& trackIds,
                   const CaloGeometry* geo,
                   const CaloTopology* caloTopology,
                   int ieta,
                   int iphi,
                   genSimInfo& info,
                   bool debug) {
#ifdef EDM_ML_DEBUG
    if (debug)
      std::cout << "eGenSimInfo:: For track " << (*trkItr)->momentum().rho() << "/" << (*trkItr)->momentum().eta()
                << "/" << (*trkItr)->momentum().phi() << " with ieta:iphi " << ieta << ":" << iphi << std::endl;
#endif
    std::vector<DetId> vdets = spr::matrixECALIds(coreDet, ieta, iphi, geo, caloTopology, false);
#ifdef EDM_ML_DEBUG
    if (debug)
      spr::debugEcalDets(0, vdets);
#endif
    spr::cGenSimInfo(vdets, trkItr, trackIds, true, info, debug);
  }

  void eGenSimInfo(const DetId& coreDet,
                   HepMC::GenEvent::particle_const_iterator trkItr,
                   std::vector<spr::propagatedGenTrackID>& trackIds,
                   const CaloGeometry* geo,
                   const CaloTopology* caloTopology,
                   double dR,
                   const GlobalVector& trackMom,
                   spr::genSimInfo& info,
                   bool debug) {
#ifdef EDM_ML_DEBUG
    if (debug)
      std::cout << "eGenSimInfo:: For track " << (*trkItr)->momentum().rho() << "/" << (*trkItr)->momentum().eta()
                << "/" << (*trkItr)->momentum().phi() << " with dR,tMom " << dR << " " << trackMom << std::endl;
#endif
    std::vector<DetId> vdets = spr::matrixECALIds(coreDet, dR, trackMom, geo, caloTopology, false);
#ifdef EDM_ML_DEBUG
    if (debug)
      spr::debugEcalDets(0, vdets);
#endif
    spr::cGenSimInfo(vdets, trkItr, trackIds, true, info, debug);
  }

  void eGenSimInfo(const DetId& coreDet,
                   reco::GenParticleCollection::const_iterator trkItr,
                   std::vector<spr::propagatedGenParticleID>& trackIds,
                   const CaloGeometry* geo,
                   const CaloTopology* caloTopology,
                   int ieta,
                   int iphi,
                   genSimInfo& info,
                   bool debug) {
#ifdef EDM_ML_DEBUG
    if (debug)
      std::cout << "eGenSimInfo:: For track " << trkItr->momentum().R() << "/" << trkItr->momentum().eta() << "/"
                << trkItr->momentum().phi() << " with ieta:iphi " << ieta << ":" << iphi << std::endl;
#endif
    std::vector<DetId> vdets = spr::matrixECALIds(coreDet, ieta, iphi, geo, caloTopology, false);
#ifdef EDM_ML_DEBUG
    if (debug)
      spr::debugEcalDets(0, vdets);
#endif
    spr::cGenSimInfo(vdets, trkItr, trackIds, true, info, debug);
  }

  void eGenSimInfo(const DetId& coreDet,
                   reco::GenParticleCollection::const_iterator trkItr,
                   std::vector<spr::propagatedGenParticleID>& trackIds,
                   const CaloGeometry* geo,
                   const CaloTopology* caloTopology,
                   double dR,
                   const GlobalVector& trackMom,
                   spr::genSimInfo& info,
                   bool debug) {
#ifdef EDM_ML_DEBUG
    if (debug)
      std::cout << "eGenSimInfo:: For track " << trkItr->momentum().R() << "/" << trkItr->momentum().eta() << "/"
                << trkItr->momentum().phi() << " with dR,tMom " << dR << " " << trackMom << std::endl;
#endif
    std::vector<DetId> vdets = spr::matrixECALIds(coreDet, dR, trackMom, geo, caloTopology, false);
#ifdef EDM_ML_DEBUG
    if (debug)
      spr::debugEcalDets(0, vdets);
#endif
    spr::cGenSimInfo(vdets, trkItr, trackIds, true, info, debug);
  }

  void hGenSimInfo(const DetId& coreDet,
                   HepMC::GenEvent::particle_const_iterator trkItr,
                   std::vector<spr::propagatedGenTrackID>& trackIds,
                   const HcalTopology* topology,
                   int ieta,
                   int iphi,
                   genSimInfo& info,
                   bool includeHO,
                   bool debug) {
#ifdef EDM_ML_DEBUG
    if (debug)
      std::cout << "hGenSimInfo:: For track " << (*trkItr)->momentum().rho() << "/" << (*trkItr)->momentum().eta()
                << "/" << (*trkItr)->momentum().phi() << " with ieta:iphi " << ieta << ":" << iphi << std::endl;
#endif
    std::vector<DetId> dets;
    dets.push_back(coreDet);
    std::vector<DetId> vdets = spr::matrixHCALIds(dets, topology, ieta, iphi, includeHO, false);
#ifdef EDM_ML_DEBUG
    if (debug)
      spr::debugHcalDets(0, vdets);
#endif
    spr::cGenSimInfo(vdets, trkItr, trackIds, false, info, debug);
  }

  void hGenSimInfo(const DetId& coreDet,
                   HepMC::GenEvent::particle_const_iterator trkItr,
                   std::vector<spr::propagatedGenTrackID>& trackIds,
                   const CaloGeometry* geo,
                   const HcalTopology* topology,
                   double dR,
                   const GlobalVector& trackMom,
                   spr::genSimInfo& info,
                   bool includeHO,
                   bool debug) {
#ifdef EDM_ML_DEBUG
    if (debug)
      std::cout << "hGenSimInfo:: For track " << (*trkItr)->momentum().rho() << "/" << (*trkItr)->momentum().eta()
                << "/" << (*trkItr)->momentum().phi() << " with dR,tMom " << dR << " " << trackMom << std::endl;
#endif
    std::vector<DetId> vdets = spr::matrixHCALIds(coreDet, geo, topology, dR, trackMom, includeHO, false);
#ifdef EDM_ML_DEBUG
    if (debug)
      spr::debugHcalDets(0, vdets);
#endif
    spr::cGenSimInfo(vdets, trkItr, trackIds, false, info, debug);
  }

  void hGenSimInfo(const DetId& coreDet,
                   reco::GenParticleCollection::const_iterator trkItr,
                   std::vector<spr::propagatedGenParticleID>& trackIds,
                   const HcalTopology* topology,
                   int ieta,
                   int iphi,
                   genSimInfo& info,
                   bool includeHO,
                   bool debug) {
#ifdef EDM_ML_DEBUG
    if (debug)
      std::cout << "hGenSimInfo:: For track " << trkItr->momentum().R() << "/" << trkItr->momentum().eta() << "/"
                << trkItr->momentum().phi() << " with ieta:iphi " << ieta << ":" << iphi << std::endl;
#endif
    std::vector<DetId> dets;
    dets.push_back(coreDet);
    std::vector<DetId> vdets = spr::matrixHCALIds(dets, topology, ieta, iphi, includeHO, false);
#ifdef EDM_ML_DEBUG
    if (debug)
      spr::debugHcalDets(0, vdets);
#endif
    spr::cGenSimInfo(vdets, trkItr, trackIds, false, info, debug);
  }

  void hGenSimInfo(const DetId& coreDet,
                   reco::GenParticleCollection::const_iterator trkItr,
                   std::vector<spr::propagatedGenParticleID>& trackIds,
                   const CaloGeometry* geo,
                   const HcalTopology* topology,
                   double dR,
                   const GlobalVector& trackMom,
                   spr::genSimInfo& info,
                   bool includeHO,
                   bool debug) {
#ifdef EDM_ML_DEBUG
    if (debug)
      std::cout << "hGenSimInfo:: For track " << trkItr->momentum().R() << "/" << trkItr->momentum().eta() << "/"
                << trkItr->momentum().phi() << " with dR,tMom " << dR << " " << trackMom << std::endl;
#endif
    std::vector<DetId> vdets = spr::matrixHCALIds(coreDet, geo, topology, dR, trackMom, includeHO, false);
#ifdef EDM_ML_DEBUG
    if (debug)
      spr::debugHcalDets(0, vdets);
#endif
    spr::cGenSimInfo(vdets, trkItr, trackIds, false, info, debug);
  }

  void cGenSimInfo(std::vector<DetId>& vdets,
                   HepMC::GenEvent::particle_const_iterator trkItr,
                   std::vector<spr::propagatedGenTrackID>& trackIds,
                   bool ifECAL,
                   spr::genSimInfo& info,
                   bool debug) {
    info.maxNearP = -1.0;
    info.cHadronEne = info.nHadronEne = info.eleEne = info.muEne = info.photonEne = 0.0;
    info.isChargedIso = true;
    for (double& i : info.cHadronEne_)
      i = 0.0;
    for (auto& trackId : trackIds) {
      HepMC::GenEvent::particle_const_iterator trkItr2 = trackId.trkItr;
      // avoid the track under consideration
      if ((trkItr2 != trkItr) && trackId.ok) {
        int charge = trackId.charge;
        int pdgid = trackId.pdgId;
        double p = (*trkItr2)->momentum().rho();
        bool isolat = false;
        if (ifECAL) {
          const DetId anyCell = trackId.detIdECAL;
          isolat = spr::chargeIsolation(anyCell, vdets);
        } else {
          const DetId anyCell = trackId.detIdHCAL;
          isolat = spr::chargeIsolation(anyCell, vdets);
        }
        if (!isolat)
          spr::cGenSimInfo(charge, pdgid, p, info, debug);
      }
    }
#ifdef EDM_ML_DEBUG
    if (debug) {
      std::cout << "Isolation variables: isChargedIso :" << info.isChargedIso << " maxNearP " << info.maxNearP
                << " Energy e/mu/g/ch/nh " << info.eleEne << "," << info.muEne << "," << info.photonEne << ","
                << info.cHadronEne << "," << info.nHadronEne << " charge " << info.cHadronEne_[0] << ","
                << info.cHadronEne_[1] << "," << info.cHadronEne_[2] << std::endl;
    }
#endif
  }

  void cGenSimInfo(std::vector<DetId>& vdets,
                   reco::GenParticleCollection::const_iterator trkItr,
                   std::vector<spr::propagatedGenParticleID>& trackIds,
                   bool ifECAL,
                   spr::genSimInfo& info,
                   bool debug) {
    info.maxNearP = -1.0;
    info.cHadronEne = info.nHadronEne = info.eleEne = info.muEne = info.photonEne = 0.0;
    info.isChargedIso = true;
    for (double& i : info.cHadronEne_)
      i = 0.0;
    for (auto& trackId : trackIds) {
      reco::GenParticleCollection::const_iterator trkItr2 = trackId.trkItr;
      // avoid the track under consideration
      if ((trkItr2 != trkItr) && trackId.ok) {
        int charge = trackId.charge;
        int pdgid = trackId.pdgId;
        double p = trkItr2->momentum().R();
        bool isolat = false;
        if (ifECAL) {
          const DetId anyCell = trackId.detIdECAL;
          isolat = spr::chargeIsolation(anyCell, vdets);
        } else {
          const DetId anyCell = trackId.detIdHCAL;
          isolat = spr::chargeIsolation(anyCell, vdets);
        }
        if (!isolat)
          spr::cGenSimInfo(charge, pdgid, p, info, debug);
      }
    }

#ifdef EDM_ML_DEBUG
    if (debug) {
      std::cout << "Isolation variables: isChargedIso :" << info.isChargedIso << " maxNearP " << info.maxNearP
                << " Energy e/mu/g/ch/nh " << info.eleEne << "," << info.muEne << "," << info.photonEne << ","
                << info.cHadronEne << "," << info.nHadronEne << " charge " << info.cHadronEne_[0] << ","
                << info.cHadronEne_[1] << "," << info.cHadronEne_[2] << std::endl;
    }
#endif
  }

  void cGenSimInfo(int charge, int pdgid, double p, spr::genSimInfo& info, bool) {
    if (pdgid == 22)
      info.photonEne += p;
    else if (pdgid == 11)
      info.eleEne += p;
    else if (pdgid == 13)
      info.muEne += p;
    else if (std::abs(charge) > 0) {
      info.isChargedIso = false;
      info.cHadronEne += p;
      if (p > 1.0)
        info.cHadronEne_[0] += p;
      if (p > 2.0)
        info.cHadronEne_[1] += p;
      if (p > 3.0)
        info.cHadronEne_[2] += p;
      if (info.maxNearP < p)
        info.maxNearP = p;
    } else if (std::abs(charge) == 0) {
      info.nHadronEne += p;
    }
  }
}  // namespace spr
