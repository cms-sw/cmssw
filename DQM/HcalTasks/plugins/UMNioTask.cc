
#include "DQM/HcalTasks/interface/UMNioTask.h"

using namespace hcaldqm;
using namespace hcaldqm::constants;
UMNioTask::UMNioTask(edm::ParameterSet const& ps) : DQTask(ps) {
  _tagHBHE = ps.getUntrackedParameter<edm::InputTag>("tagHBHE", edm::InputTag("hcalDigis"));
  _tagHO = ps.getUntrackedParameter<edm::InputTag>("tagHO", edm::InputTag("hcalDigis"));
  _tagHF = ps.getUntrackedParameter<edm::InputTag>("tagHF", edm::InputTag("hcalDigis"));
  _taguMN = ps.getUntrackedParameter<edm::InputTag>("taguMN", edm::InputTag("hcalDigis"));

  _tokHBHE = consumes<HBHEDigiCollection>(_tagHBHE);
  _tokHO = consumes<HODigiCollection>(_tagHO);
  _tokHF = consumes<HFDigiCollection>(_tagHF);
  _tokuMN = consumes<HcalUMNioDigi>(_taguMN);

  _lowHBHE = ps.getUntrackedParameter<double>("lowHBHE", 20);
  _lowHO = ps.getUntrackedParameter<double>("lowHO", 20);
  _lowHF = ps.getUntrackedParameter<double>("lowHF", 20);

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

  edm::ESHandle<HcalDbService> dbService;
  es.get<HcalDbRecord>().get(dbService);
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
  constants::OrbitGapType orbitGapType;
  if (eventType == constants::EVENTTYPE_PEDESTAL) {
    orbitGapType = tPedestal;
  } else if (eventType == constants::EVENTTYPE_LED) {
    orbitGapType = tLED;
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
      //case tSafe : return "Safe";
      default:
        return tUnknown;
    }
  }
  return (int)(std::find(_eventtypes.begin(), _eventtypes.end(), orbitGapType) - _eventtypes.begin());
}

/* virtual */ void UMNioTask::_process(edm::Event const& e, edm::EventSetup const& es) {
  edm::Handle<HcalUMNioDigi> cumn;
  if (!e.getByToken(_tokuMN, cumn))
    return;

  uint8_t eventType = cumn->eventType();
  uint32_t laserType = cumn->valueUserWord(0);
  _cEventType.fill(_currentLS, getOrbitGapIndex(eventType, laserType));

  //	Compute the Total Charge in the Detector...
  edm::Handle<HBHEDigiCollection> chbhe;
  edm::Handle<HODigiCollection> cho;
  edm::Handle<HFDigiCollection> chf;

  if (!e.getByToken(_tokHBHE, chbhe))
    _logger.dqmthrow("Collection HBHEDigiCollection isn't available " + _tagHBHE.label() + " " + _tagHBHE.instance());
  if (!e.getByToken(_tokHO, cho))
    _logger.dqmthrow("Collection HODigiCollection isn't available " + _tagHO.label() + " " + _tagHO.instance());
  if (!e.getByToken(_tokHF, chf))
    _logger.dqmthrow("Collection HFDigiCollection isn't available " + _tagHF.label() + " " + _tagHF.instance());

  for (HBHEDigiCollection::const_iterator it = chbhe->begin(); it != chbhe->end(); ++it) {
    double sumQ = hcaldqm::utilities::sumQ<HBHEDataFrame>(*it, 2.5, 0, it->size() - 1);
    _cTotalCharge.fill(it->id(), _currentLS, sumQ);
    _cTotalChargeProfile.fill(it->id(), _currentLS, sumQ);
  }
  for (HODigiCollection::const_iterator it = cho->begin(); it != cho->end(); ++it) {
    double sumQ = hcaldqm::utilities::sumQ<HODataFrame>(*it, 8.5, 0, it->size() - 1);
    _cTotalCharge.fill(it->id(), _currentLS, sumQ);
    _cTotalChargeProfile.fill(it->id(), _currentLS, sumQ);
  }
  for (HFDigiCollection::const_iterator it = chf->begin(); it != chf->end(); ++it) {
    double sumQ = hcaldqm::utilities::sumQ<HFDataFrame>(*it, 2.5, 0, it->size() - 1);
    _cTotalCharge.fill(it->id(), _currentLS, sumQ);
    _cTotalChargeProfile.fill(it->id(), _currentLS, sumQ);
  }
}
/* virtual */ void UMNioTask::dqmEndLuminosityBlock(edm::LuminosityBlock const& lb, edm::EventSetup const& es) {
  DQTask::dqmEndLuminosityBlock(lb, es);
}

DEFINE_FWK_MODULE(UMNioTask);
