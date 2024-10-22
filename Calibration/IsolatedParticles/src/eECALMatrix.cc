#include "Calibration/IsolatedParticles/interface/eECALMatrix.h"
#include "Calibration/IsolatedParticles/interface/FindCaloHit.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"

#include <sstream>

namespace spr {

  std::pair<double, bool> energyECAL(const DetId& id,
                                     edm::Handle<EcalRecHitCollection>& hitsEC,
                                     const EcalSeverityLevelAlgo* sevlv,
                                     bool testSpike,
                                     double tMin,
                                     double tMax,
                                     bool debug) {
    std::vector<EcalRecHitCollection::const_iterator> hits;
    spr::findHit(hitsEC, id, hits, debug);

    std::ostringstream st1;
    if (debug)
      st1 << "Xtal 0x" << std::hex << id() << std::dec;
    const EcalRecHitCollection* recHitsEC = (hitsEC.isValid()) ? hitsEC.product() : nullptr;
    bool flag = (!testSpike) ? true : (sevlv->severityLevel(id, (*recHitsEC)) != EcalSeverityLevel::kWeird);
    double ener(0);
    for (const auto& hit : hits) {
      double en(0), tt(0);
      if (hit != hitsEC->end()) {
        en = hit->energy();
        tt = hit->time();
      }
      if (debug)
        st1 << " " << tt << " " << en;
      if (tt > tMin && tt < tMax)
        ener += en;
    }
    if (debug) {
      if (!flag)
        st1 << " detected to be a spike";
      edm::LogVerbatim("IsoTrack") << st1.str();
    }
    return std::pair<double, bool>(ener, flag);
  }

  std::pair<double, bool> energyECAL(const std::vector<DetId>& vdets,
                                     edm::Handle<EcalRecHitCollection>& hitsEC,
                                     const EcalSeverityLevelAlgo* sevlv,
                                     bool noThrCut,
                                     bool testSpike,
                                     double eThr,
                                     double tMin,
                                     double tMax,
                                     bool debug) {
    bool flag(true);
    double energySum(0.0);
    for (const auto& id : vdets) {
      if (id != DetId(0)) {
        std::pair<double, bool> ecalEn = spr::energyECAL(id, hitsEC, sevlv, testSpike, tMin, tMax, debug);
        if (!ecalEn.second)
          flag = false;
        if ((ecalEn.first > eThr) || noThrCut)
          energySum += ecalEn.first;
      }
    }
    if (debug)
      edm::LogVerbatim("IsoTrack") << "energyECAL: energySum = " << energySum << " flag = " << flag;
    return std::pair<double, bool>(energySum, flag);
  }

  std::pair<double, bool> eECALmatrix(const DetId& detId,
                                      edm::Handle<EcalRecHitCollection>& hitsEB,
                                      edm::Handle<EcalRecHitCollection>& hitsEE,
                                      const EcalChannelStatus& chStatus,
                                      const CaloGeometry* geo,
                                      const CaloTopology* caloTopology,
                                      const EcalSeverityLevelAlgo* sevlv,
                                      int ieta,
                                      int iphi,
                                      double ebThr,
                                      double eeThr,
                                      double tMin,
                                      double tMax,
                                      bool debug) {
    std::vector<DetId> vdets;
    spr::matrixECALIds(detId, ieta, iphi, geo, caloTopology, vdets, debug);
    if (debug)
      edm::LogVerbatim("IsoTrack") << "Inside eECALmatrix " << 2 * ieta + 1 << "X" << 2 * iphi + 1 << " nXtals "
                                   << vdets.size();
    bool flag(true);
    for (const auto& id : vdets) {
      if ((id.det() == DetId::Ecal) && (id.subdetId() == EcalBarrel)) {
        if (sevlv->severityLevel(id, (*hitsEB)) == EcalSeverityLevel::kWeird)
          flag = false;
      } else if ((id.det() == DetId::Ecal) && (id.subdetId() == EcalEndcap)) {
        if (sevlv->severityLevel(id, (*hitsEE)) == EcalSeverityLevel::kWeird)
          flag = false;
      }
    }
    return std::pair<double, bool>(spr::energyECAL(vdets, hitsEB, hitsEE, ebThr, eeThr, tMin, tMax, debug), flag);
  }

  std::pair<double, bool> eECALmatrix(const DetId& detId,
                                      edm::Handle<EcalRecHitCollection>& hitsEB,
                                      edm::Handle<EcalRecHitCollection>& hitsEE,
                                      const EcalChannelStatus& chStatus,
                                      const CaloGeometry* geo,
                                      const CaloTopology* caloTopology,
                                      const EcalSeverityLevelAlgo* sevlv,
                                      const EcalTrigTowerConstituentsMap& ttMap,
                                      int ieta,
                                      int iphi,
                                      double ebThr,
                                      double eeThr,
                                      double tMin,
                                      double tMax,
                                      bool debug) {
    std::vector<DetId> vdets;
    spr::matrixECALIds(detId, ieta, iphi, geo, caloTopology, vdets, debug);
    if (debug)
      edm::LogVerbatim("IsoTrack") << "Inside eECALmatrix " << 2 * ieta + 1 << "X" << 2 * iphi + 1 << " nXtals "
                                   << vdets.size();

    bool flag(true);
    double energySum = 0.0;
    for (const auto& id : vdets) {
      if ((id != DetId(0)) && (id.det() == DetId::Ecal) &&
          ((id.subdetId() == EcalBarrel) || (id.subdetId() == EcalEndcap))) {
        double eTower = spr::energyECALTower(id, hitsEB, hitsEE, ttMap, debug);
        bool ok = (id.subdetId() == EcalBarrel) ? (eTower > ebThr) : (eTower > eeThr);
        if (debug && (!ok))
          edm::LogVerbatim("IsoTrack") << "Crystal 0x" << std::hex << id() << std::dec << " Flag " << ok;
        if (ok) {
          std::pair<double, bool> ecalEn = (id.subdetId() == EcalBarrel)
                                               ? spr::energyECAL(id, hitsEB, sevlv, true, tMin, tMax, debug)
                                               : spr::energyECAL(id, hitsEE, sevlv, false, tMin, tMax, debug);
          if (!ecalEn.second)
            flag = false;
          energySum += ecalEn.first;
        }
      }
    }
    if (debug)
      edm::LogVerbatim("IsoTrack") << "energyECAL: energySum = " << energySum << " flag = " << flag;
    return std::pair<double, bool>(energySum, flag);
  }

  std::pair<double, bool> eECALmatrix(const HcalDetId& detId,
                                      edm::Handle<EcalRecHitCollection>& hitsEB,
                                      edm::Handle<EcalRecHitCollection>& hitsEE,
                                      const CaloGeometry* geo,
                                      const CaloTowerConstituentsMap* ctmap,
                                      const EcalSeverityLevelAlgo* sevlv,
                                      double ebThr,
                                      double eeThr,
                                      double tMin,
                                      double tMax,
                                      bool debug) {
    CaloTowerDetId tower = ctmap->towerOf(detId);
    std::vector<DetId> ids = ctmap->constituentsOf(tower);
    if (debug) {
      edm::LogVerbatim("IsoTrack") << "eECALmatrix: " << detId << " belongs to " << tower << " which has " << ids.size()
                                   << " constituents";
      for (unsigned int i = 0; i < ids.size(); ++i) {
        std::ostringstream st1;
        st1 << "[" << i << "] " << std::hex << ids[i].rawId() << std::dec;
        if (ids[i].det() == DetId::Ecal && ids[i].subdetId() == EcalBarrel) {
          st1 << " " << EBDetId(ids[i]);
        } else if (ids[i].det() == DetId::Ecal && ids[i].subdetId() == EcalEndcap) {
          st1 << " " << EEDetId(ids[i]);
        } else if (ids[i].det() == DetId::Ecal && ids[i].subdetId() == EcalPreshower) {
          st1 << " " << ESDetId(ids[i]);
        } else if (ids[i].det() == DetId::Hcal) {
          st1 << " " << HcalDetId(ids[i]);
        }
        edm::LogVerbatim("IsoTrack") << st1.str();
      }
    }

    std::vector<DetId> idEBEE;
    bool flag(true);
    for (const auto& id : ids) {
      if ((id.det() == DetId::Ecal) && (id.subdetId() == EcalBarrel)) {
        idEBEE.emplace_back(id);
        if (sevlv->severityLevel(id, (*hitsEB)) == EcalSeverityLevel::kWeird)
          flag = false;
      } else if ((id.det() == DetId::Ecal) && (id.subdetId() == EcalEndcap)) {
        idEBEE.emplace_back(id);
        if (sevlv->severityLevel(id, (*hitsEE)) == EcalSeverityLevel::kWeird)
          flag = false;
      }
    }

    if (debug)
      edm::LogVerbatim("IsoTrack") << "eECALmatrix: with " << idEBEE.size() << " EB+EE hits and "
                                   << "spike flag " << flag;
    double etot = (!idEBEE.empty()) ? spr::energyECAL(idEBEE, hitsEB, hitsEE, ebThr, eeThr, tMin, tMax, debug) : 0;
    return std::pair<double, bool>(etot, flag);
  }

  std::pair<double, bool> eECALmatrix(const DetId& detId,
                                      edm::Handle<EcalRecHitCollection>& hitsEB,
                                      edm::Handle<EcalRecHitCollection>& hitsEE,
                                      const EcalChannelStatus& chStatus,
                                      const CaloGeometry* geo,
                                      const CaloTopology* caloTopology,
                                      const EcalSeverityLevelAlgo* sevlv,
                                      const EcalPFRecHitThresholds* eThresholds,
                                      int ieta,
                                      int iphi,
                                      bool debug) {
    std::vector<DetId> vdets;
    spr::matrixECALIds(detId, ieta, iphi, geo, caloTopology, vdets, debug);
    if (debug)
      edm::LogVerbatim("IsoTrack") << "Inside eECALmatrix " << 2 * ieta + 1 << "X" << 2 * iphi + 1 << " nXtals "
                                   << vdets.size();
    bool flag(true);
    for (const auto& id : vdets) {
      if ((id.det() == DetId::Ecal) && (id.subdetId() == EcalBarrel)) {
        if (sevlv->severityLevel(id, (*hitsEB)) == EcalSeverityLevel::kWeird)
          flag = false;
      } else if ((id.det() == DetId::Ecal) && (id.subdetId() == EcalEndcap)) {
        if (sevlv->severityLevel(id, (*hitsEE)) == EcalSeverityLevel::kWeird)
          flag = false;
      }
    }
    double energySum = 0.0;
    for (unsigned int i1 = 0; i1 < vdets.size(); i1++) {
      if (vdets[i1] != DetId(0)) {
        std::vector<EcalRecHitCollection::const_iterator> hit;
        if (vdets[i1].subdetId() == EcalBarrel) {
          spr::findHit(hitsEB, vdets[i1], hit, debug);
        } else if (vdets[i1].subdetId() == EcalEndcap) {
          spr::findHit(hitsEE, vdets[i1], hit, debug);
        }
        std::ostringstream st1;
        if (debug)
          st1 << "Crystal 0x" << std::hex << vdets[i1]() << std::dec;
        double ener = 0, ethr = static_cast<double>((*eThresholds)[vdets[i1]]);
        for (unsigned int ihit = 0; ihit < hit.size(); ihit++) {
          double en = 0;
          if (vdets[i1].subdetId() == EcalBarrel) {
            if (hit[ihit] != hitsEB->end())
              en = hit[ihit]->energy();
          } else if (vdets[i1].subdetId() == EcalEndcap) {
            if (hit[ihit] != hitsEE->end())
              en = hit[ihit]->energy();
          }
          if (debug)
            st1 << " " << ihit << " " << en << " Thr " << ethr;
          ener += en;
        }
        if (debug)
          edm::LogVerbatim("IsoTrack") << st1.str();
        if (ener > ethr)
          energySum += ener;
      }
    }
    return std::pair<double, bool>(energySum, flag);
  }

}  // namespace spr
