
#include "DQM/HcalTasks/interface/HFRaddamTask.h"

using namespace hcaldqm;
using namespace hcaldqm::constants;
using namespace hcaldqm::filter;

HFRaddamTask::HFRaddamTask(edm::ParameterSet const& ps) : DQTask(ps) {
  //	List all the DetIds
  _vDetIds.push_back(HcalDetId(HcalForward, -30, 35, 1));
  _vDetIds.push_back(HcalDetId(HcalForward, -30, 71, 1));
  _vDetIds.push_back(HcalDetId(HcalForward, -32, 15, 1));
  _vDetIds.push_back(HcalDetId(HcalForward, -32, 51, 1));
  _vDetIds.push_back(HcalDetId(HcalForward, -34, 35, 1));
  _vDetIds.push_back(HcalDetId(HcalForward, -34, 71, 1));
  _vDetIds.push_back(HcalDetId(HcalForward, -36, 15, 1));
  _vDetIds.push_back(HcalDetId(HcalForward, -36, 51, 1));
  _vDetIds.push_back(HcalDetId(HcalForward, -38, 35, 1));
  _vDetIds.push_back(HcalDetId(HcalForward, -38, 71, 1));
  _vDetIds.push_back(HcalDetId(HcalForward, -40, 15, 1));
  _vDetIds.push_back(HcalDetId(HcalForward, -40, 51, 1));
  _vDetIds.push_back(HcalDetId(HcalForward, -41, 35, 1));
  _vDetIds.push_back(HcalDetId(HcalForward, -41, 71, 1));
  _vDetIds.push_back(HcalDetId(HcalForward, -30, 15, 2));
  _vDetIds.push_back(HcalDetId(HcalForward, -30, 51, 2));
  _vDetIds.push_back(HcalDetId(HcalForward, -32, 35, 2));
  _vDetIds.push_back(HcalDetId(HcalForward, -32, 71, 2));
  _vDetIds.push_back(HcalDetId(HcalForward, -34, 15, 2));
  _vDetIds.push_back(HcalDetId(HcalForward, -34, 51, 2));
  _vDetIds.push_back(HcalDetId(HcalForward, -36, 35, 2));
  _vDetIds.push_back(HcalDetId(HcalForward, -36, 71, 2));
  _vDetIds.push_back(HcalDetId(HcalForward, -38, 15, 2));
  _vDetIds.push_back(HcalDetId(HcalForward, -38, 51, 2));
  _vDetIds.push_back(HcalDetId(HcalForward, -40, 35, 2));
  _vDetIds.push_back(HcalDetId(HcalForward, -40, 71, 2));
  _vDetIds.push_back(HcalDetId(HcalForward, -41, 15, 2));
  _vDetIds.push_back(HcalDetId(HcalForward, -41, 51, 2));

  _vDetIds.push_back(HcalDetId(HcalForward, 30, 21, 1));
  _vDetIds.push_back(HcalDetId(HcalForward, 30, 57, 1));
  _vDetIds.push_back(HcalDetId(HcalForward, 32, 1, 1));
  _vDetIds.push_back(HcalDetId(HcalForward, 32, 37, 1));
  _vDetIds.push_back(HcalDetId(HcalForward, 34, 21, 1));
  _vDetIds.push_back(HcalDetId(HcalForward, 34, 57, 1));
  _vDetIds.push_back(HcalDetId(HcalForward, 36, 1, 1));
  _vDetIds.push_back(HcalDetId(HcalForward, 36, 37, 1));
  _vDetIds.push_back(HcalDetId(HcalForward, 38, 21, 1));
  _vDetIds.push_back(HcalDetId(HcalForward, 38, 57, 1));
  _vDetIds.push_back(HcalDetId(HcalForward, 40, 35, 1));
  _vDetIds.push_back(HcalDetId(HcalForward, 40, 71, 1));
  _vDetIds.push_back(HcalDetId(HcalForward, 41, 19, 1));
  _vDetIds.push_back(HcalDetId(HcalForward, 41, 55, 1));
  _vDetIds.push_back(HcalDetId(HcalForward, 30, 1, 2));
  _vDetIds.push_back(HcalDetId(HcalForward, 30, 37, 2));
  _vDetIds.push_back(HcalDetId(HcalForward, 32, 21, 2));
  _vDetIds.push_back(HcalDetId(HcalForward, 32, 57, 2));
  _vDetIds.push_back(HcalDetId(HcalForward, 34, 1, 2));
  _vDetIds.push_back(HcalDetId(HcalForward, 34, 37, 2));
  _vDetIds.push_back(HcalDetId(HcalForward, 36, 21, 2));
  _vDetIds.push_back(HcalDetId(HcalForward, 36, 57, 2));
  _vDetIds.push_back(HcalDetId(HcalForward, 38, 1, 2));
  _vDetIds.push_back(HcalDetId(HcalForward, 38, 37, 2));
  _vDetIds.push_back(HcalDetId(HcalForward, 40, 19, 2));
  _vDetIds.push_back(HcalDetId(HcalForward, 40, 55, 2));
  _vDetIds.push_back(HcalDetId(HcalForward, 41, 35, 2));
  _vDetIds.push_back(HcalDetId(HcalForward, 41, 71, 2));

  //	tags
  _tagHF = ps.getUntrackedParameter<edm::InputTag>("tagHF", edm::InputTag("hcalDigis"));
  _taguMN = ps.getUntrackedParameter<edm::InputTag>("taguMN", edm::InputTag("hcalDigis"));
  _tokHF = consumes<QIE10DigiCollection>(_tagHF);
  _tokuMN = consumes<HcalUMNioDigi>(_taguMN);
}

/* virtual */ void HFRaddamTask::bookHistograms(DQMStore::IBooker& ib, edm::Run const& r, edm::EventSetup const& es) {
  //	Initialize all the Single Containers
  for (std::vector<HcalDetId>::const_iterator it = _vDetIds.begin(); it != _vDetIds.end(); ++it) {
    _vcShape.push_back(ContainerSingle1D(_name,
                                         "Shape",
                                         new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS),
                                         new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::ffC_3000)));
  }

  DQTask::bookHistograms(ib, r, es);
  char aux[200];
  for (unsigned int i = 0; i < _vDetIds.size(); i++) {
    sprintf(aux, "ieta%diphi%dd%d", _vDetIds[i].ieta(), _vDetIds[i].iphi(), _vDetIds[i].depth());
    _vcShape[i].book(ib, _subsystem, aux);
  }
}

/* virtual */ void HFRaddamTask::_process(edm::Event const& e, edm::EventSetup const& es) {
  auto const chf = e.getHandle(_tokHF);
  if (not(chf.isValid())) {
    edm::LogWarning("HFRaddamTask") << "QIE10 collection not valid for HF";
    return;
  }

  for (QIE10DigiCollection::const_iterator it = chf->begin(); it != chf->end(); ++it) {
    const QIE10DataFrame digi = static_cast<const QIE10DataFrame>(*it);
    HcalDetId const& did = digi.detid();
    if (did.subdet() != HcalForward)
      continue;

    CaloSamples digi_fC = hcaldqm::utilities::loadADC2fCDB<QIE10DataFrame>(_dbService, did, digi);

    for (unsigned int i = 0; i < _vDetIds.size(); i++)
      if (did == _vDetIds[i]) {
        for (int j = 0; j < digi.samples(); j++) {
          double q = hcaldqm::utilities::adc2fCDBMinusPedestal<QIE10DataFrame>(_dbService, digi_fC, did, digi, j);
          _vcShape[i].fill(j, q);
        }
      }
  }
}

/* virtual */ bool HFRaddamTask::_isApplicable(edm::Event const& e) {
  if (_ptype == fOnline) {
    edm::Handle<HcalUMNioDigi> cumn;
    if (!e.getByToken(_tokuMN, cumn))
      return false;

    //  event type check
    uint8_t eventType = cumn->eventType();
    if (eventType == constants::EVENTTYPE_HFRADDAM)
      return true;
  } else if (_ptype == fLocal) {
    //	local, just return true as all the settings will be done in cfg
    return true;
  }

  return false;
}

DEFINE_FWK_MODULE(HFRaddamTask);
