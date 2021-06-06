#include "RecoTauTag/RecoTau/interface/AntiElectronDeadECAL.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

AntiElectronDeadECAL::AntiElectronDeadECAL(const edm::ParameterSet& cfg, edm::ConsumesCollector&& cc)
    : minStatus_(cfg.getParameter<uint32_t>("minStatus")),
      dR2_(std::pow(cfg.getParameter<double>("dR"), 2)),
      extrapolateToECalEntrance_(cfg.getParameter<bool>("extrapolateToECalEntrance")),
      verbosity_(cfg.getParameter<int>("verbosity")),
      positionAtECalEntrance_(PositionAtECalEntranceComputer(cc)),
      isFirstEvent_(true) {}

AntiElectronDeadECAL::~AntiElectronDeadECAL() {}

void AntiElectronDeadECAL::beginEvent(const edm::EventSetup& es) {
  updateBadTowers(es);
  positionAtECalEntrance_.beginEvent(es);
}

namespace {
  template <class Id>
  void loopXtals(std::map<uint32_t, unsigned>& nBadCrystals,
                 std::map<uint32_t, unsigned>& maxStatus,
                 std::map<uint32_t, double>& sumEta,
                 std::map<uint32_t, double>& sumPhi,
                 const EcalChannelStatus* channelStatus,
                 const CaloGeometry* caloGeometry,
                 const EcalTrigTowerConstituentsMap* ttMap,
                 unsigned minStatus,
                 const uint16_t statusMask) {
    // NOTE: modified version of SUSY CAF code
    //         UserCode/SusyCAF/plugins/SusyCAF_EcalDeadChannels.cc
    for (int i = 0; i < Id::kSizeForDenseIndexing; ++i) {
      Id id = Id::unhashIndex(i);
      if (id == Id(0))
        continue;
      EcalChannelStatusMap::const_iterator it = channelStatus->getMap().find(id.rawId());
      unsigned status = (it == channelStatus->end()) ? 0 : (it->getStatusCode() & statusMask);
      if (status >= minStatus) {
        const GlobalPoint& point = caloGeometry->getPosition(id);
        uint32_t key = ttMap->towerOf(id);
        maxStatus[key] = std::max(status, maxStatus[key]);
        ++nBadCrystals[key];
        sumEta[key] += point.eta();
        sumPhi[key] += point.phi();
      }
    }
  }
}  // namespace

void AntiElectronDeadECAL::updateBadTowers(const edm::EventSetup& es) {
  // NOTE: modified version of SUSY CAF code
  //         UserCode/SusyCAF/plugins/SusyCAF_EcalDeadChannels.cc

  if (!isFirstEvent_ && !channelStatusWatcher_.check(es) && !caloGeometryWatcher_.check(es) &&
      !idealGeometryWatcher_.check(es))
    return;

  edm::ESHandle<EcalChannelStatus> channelStatus;
  es.get<EcalChannelStatusRcd>().get(channelStatus);

  edm::ESHandle<CaloGeometry> caloGeometry;
  es.get<CaloGeometryRecord>().get(caloGeometry);

  edm::ESHandle<EcalTrigTowerConstituentsMap> ttMap;
  es.get<IdealGeometryRecord>().get(ttMap);

  std::map<uint32_t, unsigned> nBadCrystals, maxStatus;
  std::map<uint32_t, double> sumEta, sumPhi;

  loopXtals<EBDetId>(nBadCrystals,
                     maxStatus,
                     sumEta,
                     sumPhi,
                     channelStatus.product(),
                     caloGeometry.product(),
                     ttMap.product(),
                     minStatus_,
                     statusMask_);
  loopXtals<EEDetId>(nBadCrystals,
                     maxStatus,
                     sumEta,
                     sumPhi,
                     channelStatus.product(),
                     caloGeometry.product(),
                     ttMap.product(),
                     minStatus_,
                     statusMask_);

  badTowers_.clear();
  for (auto it : nBadCrystals) {
    uint32_t key = it.first;
    badTowers_.push_back(TowerInfo(key, it.second, maxStatus[key], sumEta[key] / it.second, sumPhi[key] / it.second));
  }

  isFirstEvent_ = false;
}

bool AntiElectronDeadECAL::operator()(const reco::Candidate* tau) const {
  bool isNearBadTower = false;
  double tau_eta = tau->eta();
  double tau_phi = tau->phi();
  const reco::Candidate* leadChargedHadron = nullptr;
  if (extrapolateToECalEntrance_) {
    const reco::PFTau* pfTau = dynamic_cast<const reco::PFTau*>(tau);
    if (pfTau != nullptr) {
      leadChargedHadron = pfTau->leadChargedHadrCand().isNonnull() ? pfTau->leadChargedHadrCand().get() : nullptr;
    } else {
      const pat::Tau* patTau = dynamic_cast<const pat::Tau*>(tau);
      if (patTau != nullptr) {
        leadChargedHadron = patTau->leadChargedHadrCand().isNonnull() ? patTau->leadChargedHadrCand().get() : nullptr;
      }
    }
  }
  if (leadChargedHadron != nullptr) {
    bool success = false;
    reco::Candidate::Point positionAtECalEntrance = positionAtECalEntrance_(leadChargedHadron, success);
    if (success) {
      tau_eta = positionAtECalEntrance.eta();
      tau_phi = positionAtECalEntrance.phi();
    }
  }
  if (verbosity_) {
    edm::LogPrint("TauAgainstEleDeadECAL") << "<AntiElectronDeadECal::operator()>:";
    edm::LogPrint("TauAgainstEleDeadECAL") << " #badTowers = " << badTowers_.size();
    if (leadChargedHadron != nullptr) {
      edm::LogPrint("TauAgainstEleDeadECAL")
          << " leadChargedHadron (" << leadChargedHadron->pdgId() << "): Pt = " << leadChargedHadron->pt()
          << ", eta at ECAL (vtx) = " << tau_eta << " (" << leadChargedHadron->eta() << ")"
          << ", phi at ECAL (vtx) = " << tau_phi << " (" << leadChargedHadron->phi() << ")";
    } else {
      edm::LogPrint("TauAgainstEleDeadECAL")
          << " tau: Pt = " << tau->pt() << ", eta at vtx = " << tau_eta << ", phi at vtx = " << tau_phi;
    }
  }
  for (auto const& badTower : badTowers_) {
    if (deltaR2(badTower.eta_, badTower.phi_, tau_eta, tau_phi) < dR2_) {
      if (verbosity_) {
        edm::LogPrint("TauAgainstEleDeadECAL")
            << " matches badTower: eta = " << badTower.eta_ << ", phi = " << badTower.phi_;
      }
      isNearBadTower = true;
      break;
    }
  }
  return isNearBadTower;
}
