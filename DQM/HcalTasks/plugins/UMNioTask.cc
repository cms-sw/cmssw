
#include "DQM/HcalTasks/interface/UMNioTask.h"

using namespace hcaldqm;
using namespace hcaldqm::constants;
UMNioTask::UMNioTask(edm::ParameterSet const& ps)
    : DQTask(ps), hcalDbServiceToken_(esConsumes<HcalDbService, HcalDbRecord, edm::Transition::BeginRun>()) {
  tagHBHE_ = ps.getUntrackedParameter<edm::InputTag>("tagHBHE", edm::InputTag("hcalDigis"));
  tagHO_ = ps.getUntrackedParameter<edm::InputTag>("tagHO", edm::InputTag("hcalDigis"));
  tagHF_ = ps.getUntrackedParameter<edm::InputTag>("tagHF", edm::InputTag("hcalDigis"));
  taguMN_ = ps.getUntrackedParameter<edm::InputTag>("taguMN", edm::InputTag("hcalDigis"));

  tokHBHE_ = consumes<QIE11DigiCollection>(tagHBHE_);
  tokHO_ = consumes<HODigiCollection>(tagHO_);
  tokHF_ = consumes<QIE10DigiCollection>(tagHF_);
  tokuMN_ = consumes<HcalUMNioDigi>(taguMN_);

  lowHBHE_ = ps.getUntrackedParameter<double>("lowHBHE", 20);
  lowHO_ = ps.getUntrackedParameter<double>("lowHO", 20);
  lowHF_ = ps.getUntrackedParameter<double>("lowHF", 20);

  //	push all the event types to monitor - whole range basically
  //	This corresponds to all enum values in hcaldqm::constants::OrbitGapType
  for (uint32_t type = constants::tNull; type < constants::nOrbitGapType; type++) {
    _eventtypes.push_back(type);
  }
}

/* virtual */ void UMNioTask::bookHistograms(DQMStore::IBooker& ib, edm::Run const& r, edm::EventSetup const& es) {
  if (_ptype == fLocal)
    if (r.runAuxiliary().run() == 1)
      return;

  DQTask::bookHistograms(ib, r, es);

  edm::ESHandle<HcalDbService> dbService = es.getHandle(hcalDbServiceToken_);
  _emap = dbService->getHcalMapping();

  _cEventType.initialize(_name,
                         "EventType",
                         new hcaldqm::quantity::LumiSection(_maxLS),
                         new hcaldqm::quantity::EventType(_eventtypes),
                         new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                         0);
  _cTotalCharge.initialize(_name,
                           "TotalCharge",
                           new hcaldqm::quantity::LumiSection(_maxLS),
                           new hcaldqm::quantity::DetectorQuantity(quantity::fSubdetPM),
                           new hcaldqm::quantity::ValueQuantity(quantity::ffC_10000, true),
                           0);
  _cTotalChargeProfile.initialize(_name,
                                  "TotalChargeProfile",
                                  new hcaldqm::quantity::LumiSection(_maxLS),
                                  new hcaldqm::quantity::DetectorQuantity(quantity::fSubdetPM),
                                  new hcaldqm::quantity::ValueQuantity(quantity::ffC_10000, true),
                                  0);
  _cEventType.book(ib, _subsystem);
  _cTotalCharge.book(ib, _subsystem);
  _cTotalChargeProfile.book(ib, _subsystem);
}

int UMNioTask::getOrbitGapIndex(uint8_t eventType, uint32_t laserType) {
  constants::OrbitGapType orbitGapType = tNull;
  if (eventType == constants::EVENTTYPE_PHYSICS) {
    orbitGapType = tPhysics;
  } else if (eventType == constants::EVENTTYPE_PEDESTAL) {
    orbitGapType = tPedestal;
  } else if (eventType == constants::EVENTTYPE_LED) {
    orbitGapType = tLED;
  } else if (eventType == constants::EVENTTYPE_HFRADDAM) {
    orbitGapType = tHFRaddam;
  } else if (eventType == constants::EVENTTYPE_LASER) {
    switch (laserType) {
      //case tNull : return "Null";
      //case tHFRaddam : return "HFRaddam";
      case 3:
        return tHBHEHPD;
      case 4:
        return tHO;
      case 5:
        return tHF;
      //case tZDC : return "ZDC";
      case 7:
        return tHEPMega;
      case 8:
        return tHEMMega;
      case 9:
        return tHBPMega;
      case 10:
        return tHBMMega;
      //case tCRF : return "CRF";
      //case tCalib : return "Calib";
      case 14:
        return tSafe;
      case 23:
        return tSiPMPMT;
      case 24:
        return tMegatile;
      default:
        return tUnknown;
    }
  }
  return (int)(std::find(_eventtypes.begin(), _eventtypes.end(), orbitGapType) - _eventtypes.begin());
}

/* virtual */ void UMNioTask::_process(edm::Event const& e, edm::EventSetup const& es) {
  auto lumiCache = luminosityBlockCache(e.getLuminosityBlock().index());
  _currentLS = lumiCache->currentLS;
  _xQuality.reset();
  _xQuality = lumiCache->xQuality;

  auto const cumn = e.getHandle(tokuMN_);
  if (not(cumn.isValid())) {
    edm::LogWarning("UMNioTask") << "HcalUMNioDigi isn't available, calling return";
    return;
  }

  uint8_t eventType = cumn->eventType();
  uint32_t laserType = cumn->valueUserWord(0);
  _cEventType.fill(_currentLS, getOrbitGapIndex(eventType, laserType));

  //	Compute the Total Charge in the Detector...
  auto const chbhe = e.getHandle(tokHBHE_);
  if (chbhe.isValid()) {
    for (QIE11DigiCollection::const_iterator it = chbhe->begin(); it != chbhe->end(); ++it) {
      const QIE11DataFrame digi = static_cast<const QIE11DataFrame>(*it);
      HcalDetId const& did = digi.detid();
      if ((did.subdet() != HcalBarrel) && (did.subdet() != HcalEndcap))
        continue;
      if (_xQuality.exists(did)) {
        HcalChannelStatus cs(did.rawId(), _xQuality.get(did));
        if (cs.isBitSet(HcalChannelStatus::HcalCellMask) || cs.isBitSet(HcalChannelStatus::HcalCellDead))
          continue;
      }
      CaloSamples digi_fC = hcaldqm::utilities::loadADC2fCDB<QIE11DataFrame>(_dbService, did, digi);
      double sumQ = hcaldqm::utilities::sumQDB<QIE11DataFrame>(_dbService, digi_fC, did, digi, 0, digi.samples() - 1);
      _cTotalCharge.fill(did, _currentLS, sumQ);
      _cTotalChargeProfile.fill(did, _currentLS, sumQ);
    }
  }
  auto const cho = e.getHandle(tokHO_);
  if (cho.isValid()) {
    for (HODigiCollection::const_iterator it = cho->begin(); it != cho->end(); ++it) {
      const HODataFrame digi = (const HODataFrame)(*it);
      HcalDetId did = digi.id();
      if (did.subdet() != HcalOuter)
        continue;
      if (_xQuality.exists(did)) {
        HcalChannelStatus cs(did.rawId(), _xQuality.get(did));
        if (cs.isBitSet(HcalChannelStatus::HcalCellMask) || cs.isBitSet(HcalChannelStatus::HcalCellDead))
          continue;
      }
      CaloSamples digi_fC = hcaldqm::utilities::loadADC2fCDB<HODataFrame>(_dbService, did, digi);
      double sumQ = hcaldqm::utilities::sumQDB<HODataFrame>(_dbService, digi_fC, did, digi, 0, digi.size() - 1);
      _cTotalCharge.fill(did, _currentLS, sumQ);
      _cTotalChargeProfile.fill(did, _currentLS, sumQ);
    }
  }
  auto const chf = e.getHandle(tokHF_);
  if (chf.isValid()) {
    for (QIE10DigiCollection::const_iterator it = chf->begin(); it != chf->end(); ++it) {
      const QIE10DataFrame digi = static_cast<const QIE10DataFrame>(*it);
      HcalDetId did = digi.detid();
      if (did.subdet() != HcalForward)
        continue;
      if (_xQuality.exists(did)) {
        HcalChannelStatus cs(did.rawId(), _xQuality.get(did));
        if (cs.isBitSet(HcalChannelStatus::HcalCellMask) || cs.isBitSet(HcalChannelStatus::HcalCellDead))
          continue;
      }
      CaloSamples digi_fC = hcaldqm::utilities::loadADC2fCDB<QIE10DataFrame>(_dbService, did, digi);
      double sumQ = hcaldqm::utilities::sumQDB<QIE10DataFrame>(_dbService, digi_fC, did, digi, 0, digi.samples() - 1);
      _cTotalCharge.fill(did, _currentLS, sumQ);
      _cTotalChargeProfile.fill(did, _currentLS, sumQ);
    }
  }
}

std::shared_ptr<hcaldqm::Cache> UMNioTask::globalBeginLuminosityBlock(edm::LuminosityBlock const& lb,
                                                                      edm::EventSetup const& es) const {
  return DQTask::globalBeginLuminosityBlock(lb, es);
}

/* virtual */ void UMNioTask::globalEndLuminosityBlock(edm::LuminosityBlock const& lb, edm::EventSetup const& es) {
  DQTask::globalEndLuminosityBlock(lb, es);
}

DEFINE_FWK_MODULE(UMNioTask);
