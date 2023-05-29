#include "DQM/EcalMonitorTasks/interface/PiZeroTask.h"

namespace ecaldqm {
  PiZeroTask::PiZeroTask()
      : DQWorkerTask(),
        seleXtalMinEnergy_(0.f),
        clusSeedThr_(0.f),
        clusEtaSize_(0),
        clusPhiSize_(0),
        selePtGammaOne_(0.f),
        selePtGammaTwo_(0.f),
        seleS4S9GammaOne_(0.f),
        seleS4S9GammaTwo_(0.f),
        selePtPi0_(0.f),
        selePi0Iso_(0.f),
        selePi0BeltDR_(0.f),
        selePi0BeltDeta_(0.f),
        seleMinvMaxPi0_(0.f),
        seleMinvMinPi0_(0.f),
        posCalcParameters_(edm::ParameterSet()) {}

  void PiZeroTask::setParams(edm::ParameterSet const& params) {
    // Parameters needed for pi0 finding
    seleXtalMinEnergy_ = params.getParameter<double>("seleXtalMinEnergy");

    clusSeedThr_ = params.getParameter<double>("clusSeedThr");
    clusEtaSize_ = params.getParameter<int>("clusEtaSize");
    clusPhiSize_ = params.getParameter<int>("clusPhiSize");

    selePtGammaOne_ = params.getParameter<double>("selePtGammaOne");
    selePtGammaTwo_ = params.getParameter<double>("selePtGammaTwo");
    seleS4S9GammaOne_ = params.getParameter<double>("seleS4S9GammaOne");
    seleS4S9GammaTwo_ = params.getParameter<double>("seleS4S9GammaTwo");
    selePtPi0_ = params.getParameter<double>("selePtPi0");
    selePi0Iso_ = params.getParameter<double>("selePi0Iso");
    selePi0BeltDR_ = params.getParameter<double>("selePi0BeltDR");
    selePi0BeltDeta_ = params.getParameter<double>("selePi0BeltDeta");
    seleMinvMaxPi0_ = params.getParameter<double>("seleMinvMaxPi0");
    seleMinvMinPi0_ = params.getParameter<double>("seleMinvMinPi0");

    posCalcParameters_ = params.getParameter<edm::ParameterSet>("posCalcParameters");
  }

  bool PiZeroTask::filterRunType(short const* runType) {
    for (unsigned iFED(0); iFED != ecaldqm::nDCC; iFED++) {
      if (runType[iFED] == EcalDCCHeaderBlock::COSMIC || runType[iFED] == EcalDCCHeaderBlock::MTCC ||
          runType[iFED] == EcalDCCHeaderBlock::COSMICS_GLOBAL || runType[iFED] == EcalDCCHeaderBlock::PHYSICS_GLOBAL ||
          runType[iFED] == EcalDCCHeaderBlock::COSMICS_LOCAL || runType[iFED] == EcalDCCHeaderBlock::PHYSICS_LOCAL)
        return true;
    }

    return false;
  }

  void PiZeroTask::runOnEBRecHits(EcalRecHitCollection const& hits) {
    MESet& mePi0MinvEB(MEs_.at("Pi0MinvEB"));
    MESet& mePi0Pt1EB(MEs_.at("Pi0Pt1EB"));
    MESet& mePi0Pt2EB(MEs_.at("Pi0Pt2EB"));
    MESet& mePi0PtEB(MEs_.at("Pi0PtEB"));
    MESet& mePi0IsoEB(MEs_.at("Pi0IsoEB"));

    const CaloSubdetectorTopology* topology_p;
    const CaloSubdetectorGeometry* geometry_p = GetGeometry()->getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
    const CaloSubdetectorGeometry* geometryES_p = GetGeometry()->getSubdetectorGeometry(DetId::Ecal, EcalPreshower);

    // Parameters for the position calculation:
    PositionCalc posCalculator_ = PositionCalc(posCalcParameters_);

    std::map<DetId, EcalRecHit> recHitsEB_map;

    std::vector<EcalRecHit> seeds;
    std::vector<EBDetId> usedXtals;
    seeds.clear();
    usedXtals.clear();

    int nClus = 0;
    std::vector<float> eClus;
    std::vector<float> etClus;
    std::vector<float> etaClus;
    std::vector<float> phiClus;
    std::vector<EBDetId> max_hit;
    std::vector<std::vector<EcalRecHit> > RecHitsCluster;
    std::vector<float> s4s9Clus;

    // Find cluster seeds in EB
    for (auto const& hit : hits) {
      EBDetId id(hit.id());
      double energy = hit.energy();
      if (energy > seleXtalMinEnergy_) {
        std::pair<DetId, EcalRecHit> map_entry(hit.id(), hit);
        recHitsEB_map.insert(map_entry);
      }
      if (energy > clusSeedThr_)
        seeds.push_back(hit);
    }  // EB rechits

    sort(seeds.begin(), seeds.end(), [](auto& x, auto& y) { return (x.energy() > y.energy()); });
    for (auto const& seed : seeds) {
      EBDetId seed_id = seed.id();

      bool seedAlreadyUsed = false;
      for (auto const& usedIds : usedXtals) {
        if (usedIds == seed_id) {
          seedAlreadyUsed = true;
          break;
        }
      }
      if (seedAlreadyUsed)
        continue;
      topology_p = GetTopology()->getSubdetectorTopology(DetId::Ecal, EcalBarrel);
      std::vector<DetId> clus_v = topology_p->getWindow(seed_id, clusEtaSize_, clusPhiSize_);
      std::vector<std::pair<DetId, float> > clus_used;

      std::vector<EcalRecHit> RecHitsInWindow;

      double simple_energy = 0;

      for (auto const& det : clus_v) {
        bool HitAlreadyUsed = false;
        for (auto const& usedIds : usedXtals) {
          if (usedIds == det) {
            HitAlreadyUsed = true;
            break;
          }
        }
        if (HitAlreadyUsed)
          continue;
        if (recHitsEB_map.find(det) != recHitsEB_map.end()) {
          std::map<DetId, EcalRecHit>::iterator aHit;
          aHit = recHitsEB_map.find(det);
          usedXtals.push_back(det);
          RecHitsInWindow.push_back(aHit->second);
          clus_used.push_back(std::pair<DetId, float>(det, 1.));
          simple_energy = simple_energy + aHit->second.energy();
        }
      }

      math::XYZPoint clus_pos = posCalculator_.Calculate_Location(clus_used, &hits, geometry_p, geometryES_p);
      float theta_s = 2. * atan(exp(-clus_pos.eta()));
      float p0x_s = simple_energy * sin(theta_s) * cos(clus_pos.phi());
      float p0y_s = simple_energy * sin(theta_s) * sin(clus_pos.phi());
      float et_s = sqrt(p0x_s * p0x_s + p0y_s * p0y_s);

      eClus.push_back(simple_energy);
      etClus.push_back(et_s);
      etaClus.push_back(clus_pos.eta());
      phiClus.push_back(clus_pos.phi());
      max_hit.push_back(seed_id);
      RecHitsCluster.push_back(RecHitsInWindow);

      // Compute S4/S9 variable
      // We are not sure to have 9 RecHits so need to check eta and phi:
      float s4s9_[4];
      for (int i = 0; i < 4; i++)
        s4s9_[i] = seed.energy();
      for (unsigned int j = 0; j < RecHitsInWindow.size(); j++) {
        if ((((EBDetId)RecHitsInWindow[j].id()).ieta() == seed_id.ieta() - 1 && seed_id.ieta() != 1) ||
            (seed_id.ieta() == 1 && (((EBDetId)RecHitsInWindow[j].id()).ieta() == seed_id.ieta() - 2))) {
          if (((EBDetId)RecHitsInWindow[j].id()).iphi() == seed_id.iphi() - 1 ||
              ((EBDetId)RecHitsInWindow[j].id()).iphi() - 360 == seed_id.iphi() - 1) {
            s4s9_[0] += RecHitsInWindow[j].energy();
          } else {
            if (((EBDetId)RecHitsInWindow[j].id()).iphi() == seed_id.iphi()) {
              s4s9_[0] += RecHitsInWindow[j].energy();
              s4s9_[1] += RecHitsInWindow[j].energy();
            } else {
              if (((EBDetId)RecHitsInWindow[j].id()).iphi() == seed_id.iphi() + 1 ||
                  ((EBDetId)RecHitsInWindow[j].id()).iphi() - 360 == seed_id.iphi() + 1) {
                s4s9_[1] += RecHitsInWindow[j].energy();
              }
            }
          }
        } else {
          if (((EBDetId)RecHitsInWindow[j].id()).ieta() == seed_id.ieta()) {
            if (((EBDetId)RecHitsInWindow[j].id()).iphi() == seed_id.iphi() - 1 ||
                ((EBDetId)RecHitsInWindow[j].id()).iphi() - 360 == seed_id.iphi() - 1) {
              s4s9_[0] += RecHitsInWindow[j].energy();
              s4s9_[3] += RecHitsInWindow[j].energy();
            } else {
              if (((EBDetId)RecHitsInWindow[j].id()).iphi() == seed_id.iphi() + 1 ||
                  ((EBDetId)RecHitsInWindow[j].id()).iphi() - 360 == seed_id.iphi() + 1) {
                s4s9_[1] += RecHitsInWindow[j].energy();
                s4s9_[2] += RecHitsInWindow[j].energy();
              }
            }
          } else {
            if ((((EBDetId)RecHitsInWindow[j].id()).ieta() == seed_id.ieta() + 1 && seed_id.ieta() != -1) ||
                (seed_id.ieta() == -1 && (((EBDetId)RecHitsInWindow[j].id()).ieta() == seed_id.ieta() + 2))) {
              if (((EBDetId)RecHitsInWindow[j].id()).iphi() == seed_id.iphi() - 1 ||
                  ((EBDetId)RecHitsInWindow[j].id()).iphi() - 360 == seed_id.iphi() - 1) {
                s4s9_[3] += RecHitsInWindow[j].energy();
              } else {
                if (((EBDetId)RecHitsInWindow[j].id()).iphi() == seed_id.iphi()) {
                  s4s9_[2] += RecHitsInWindow[j].energy();
                  s4s9_[3] += RecHitsInWindow[j].energy();
                } else {
                  if (((EBDetId)RecHitsInWindow[j].id()).iphi() == seed_id.iphi() + 1 ||
                      ((EBDetId)RecHitsInWindow[j].id()).iphi() - 360 == seed_id.iphi() + 1) {
                    s4s9_[2] += RecHitsInWindow[j].energy();
                  }
                }
              }
            } else {
              edm::LogWarning("EcalDQM") << " (EBDetId)RecHitsInWindow[j].id()).ieta() "
                                         << ((EBDetId)RecHitsInWindow[j].id()).ieta() << " seed_id.ieta() "
                                         << seed_id.ieta() << "\n"
                                         << " Problem with S4 calculation\n";
              return;
            }
          }
        }
      }
      s4s9Clus.push_back(*std::max_element(s4s9_, s4s9_ + 4) / simple_energy);
      nClus++;
      if (nClus == MAXCLUS)
        return;
    }  //  End loop over seed clusters

    // Selection, based on simple clustering
    // pi0 candidates
    int npi0_s = 0;

    std::vector<EBDetId> scXtals;
    scXtals.clear();

    if (nClus <= 1)
      return;
    for (Int_t i = 0; i < nClus; i++) {
      for (Int_t j = i + 1; j < nClus; j++) {
        if (etClus[i] > selePtGammaOne_ && etClus[j] > selePtGammaTwo_ && s4s9Clus[i] > seleS4S9GammaOne_ &&
            s4s9Clus[j] > seleS4S9GammaTwo_) {
          float theta_0 = 2. * atan(exp(-etaClus[i]));
          float theta_1 = 2. * atan(exp(-etaClus[j]));

          float p0x = eClus[i] * sin(theta_0) * cos(phiClus[i]);
          float p1x = eClus[j] * sin(theta_1) * cos(phiClus[j]);
          float p0y = eClus[i] * sin(theta_0) * sin(phiClus[i]);
          float p1y = eClus[j] * sin(theta_1) * sin(phiClus[j]);
          float p0z = eClus[i] * cos(theta_0);
          float p1z = eClus[j] * cos(theta_1);

          float pt_pi0 = sqrt((p0x + p1x) * (p0x + p1x) + (p0y + p1y) * (p0y + p1y));
          if (pt_pi0 < selePtPi0_)
            continue;
          float m_inv = sqrt((eClus[i] + eClus[j]) * (eClus[i] + eClus[j]) - (p0x + p1x) * (p0x + p1x) -
                             (p0y + p1y) * (p0y + p1y) - (p0z + p1z) * (p0z + p1z));
          if ((m_inv < seleMinvMaxPi0_) && (m_inv > seleMinvMinPi0_)) {
            // New Loop on cluster to measure isolation:
            std::vector<int> IsoClus;
            IsoClus.clear();
            float Iso = 0;
            TVector3 pi0vect = TVector3((p0x + p1x), (p0y + p1y), (p0z + p1z));
            for (Int_t k = 0; k < nClus; k++) {
              if (k == i || k == j)
                continue;
              TVector3 Clusvect = TVector3(eClus[k] * sin(2. * atan(exp(-etaClus[k]))) * cos(phiClus[k]),
                                           eClus[k] * sin(2. * atan(exp(-etaClus[k]))) * sin(phiClus[k]),
                                           eClus[k] * cos(2. * atan(exp(-etaClus[k]))));
              float dretaclpi0 = fabs(etaClus[k] - pi0vect.Eta());
              float drclpi0 = Clusvect.DeltaR(pi0vect);

              if ((drclpi0 < selePi0BeltDR_) && (dretaclpi0 < selePi0BeltDeta_)) {
                Iso = Iso + etClus[k];
                IsoClus.push_back(k);
              }
            }

            if (Iso / pt_pi0 < selePi0Iso_) {
              mePi0MinvEB.fill(getEcalDQMSetupObjects(), m_inv);
              mePi0Pt1EB.fill(getEcalDQMSetupObjects(), etClus[i]);
              mePi0Pt2EB.fill(getEcalDQMSetupObjects(), etClus[j]);
              mePi0PtEB.fill(getEcalDQMSetupObjects(), pt_pi0);
              mePi0IsoEB.fill(getEcalDQMSetupObjects(), Iso / pt_pi0);

              npi0_s++;
            }

            if (npi0_s == MAXPI0S)
              return;
          }  // pi0 inv mass window
        }    // pt and S4S9 cut
      }      // cluster "j" index loop
    }        // cluster "i" index loop
  }          // runonEBRecHits()

  DEFINE_ECALDQM_WORKER(PiZeroTask);
}  // namespace ecaldqm
