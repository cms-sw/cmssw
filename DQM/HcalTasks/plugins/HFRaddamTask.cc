
#include "DQM/HcalTasks/interface/HFRaddamTask.h"

using namespace hcaldqm;
using namespace hcaldqm::constants;
using namespace hcaldqm::filter;

HFRaddamTask::HFRaddamTask(edm::ParameterSet const& ps)
    : DQTask(ps), hcalDbServiceToken_(esConsumes<HcalDbService, HcalDbRecord, edm::Transition::BeginRun>()) {
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

  _tagFEDs = ps.getUntrackedParameter<edm::InputTag>("tagFEDs", edm::InputTag("hltHcalCalibrationRaw"));
  _tokFEDs = consumes<FEDRawDataCollection>(_tagFEDs);

  _laserType = (uint32_t)ps.getUntrackedParameter<uint32_t>("laserType");
  _nevents = ps.getUntrackedParameter<int>("nevents", 2000);
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

  // Book Raddam monitoring containers
  if (_ptype == fOnline) {
    _Raddam_ADCvsTS.initialize(_name + "/CU_Raddam",
                               "CU_Raddam_ADCvsTS",
                               new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS),
                               new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE10ADC_256),
                               new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                               0);
    _Raddam_ADCvsTS.book(ib, _subsystem);
  } else if (_ptype == fLocal) {
    _Raddam_ADCvsEvn.initialize(_name + "/CU_Raddam",
                                "CU_Raddam_ADCvsEvn",
                                new hcaldqm::quantity::EventNumber(_nevents),
                                new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fADC_256_4),
                                new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                                0);
    _Raddam_ADCvsEvn.book(ib, _subsystem);
  }

  // Extract Raddam calibration channels from emap
  edm::ESHandle<HcalDbService> dbService = es.getHandle(hcalDbServiceToken_);
  _emap = dbService->getHcalMapping();
  std::vector<HcalElectronicsId> eids = _emap->allElectronicsId();
  for (unsigned i = 0; i < eids.size(); i++) {
    HcalElectronicsId eid = eids[i];
    DetId id = _emap->lookup(eid);
    if (HcalGenericDetId(id.rawId()).isHcalCalibDetId()) {
      HcalCalibDetId calibId(id);
      if (calibId.calibFlavor() == HcalCalibDetId::CalibrationBox) {
        auto cUch = calibId.cboxChannel();
        bool isRAD(false);
        HcalSubdetector this_subdet = HcalEmpty;

        switch (calibId.hcalSubdet()) {
          case HcalBarrel:
            this_subdet = HcalBarrel;
            break;
          case HcalEndcap:
            this_subdet = HcalEndcap;
            break;
          case HcalOuter:
            this_subdet = HcalOuter;
            break;
          case HcalForward:
            this_subdet = HcalForward;
            if (cUch == 9) {
              isRAD = true;
            }
            break;
          default:
            this_subdet = HcalEmpty;
            break;
        }

        if (isRAD) {
          _raddamCalibrationChannels[this_subdet].push_back(HcalDetId(id.rawId()));
        }
      }
    }
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
    if (did.subdet() != HcalForward) {
      // Raddam monitoring from calibration channels
      if (did.subdet() == HcalOther) {
        HcalOtherDetId hodid(digi.detid());
        if (hodid.subdet() == HcalCalibration) {
          if (std::find(_raddamCalibrationChannels[HcalForward].begin(),
                        _raddamCalibrationChannels[HcalForward].end(),
                        did) != _raddamCalibrationChannels[HcalForward].end()) {
            for (int i = 0; i < digi.samples(); i++) {
              if (_ptype == fOnline) {
                _Raddam_ADCvsTS.fill(i, digi[i].adc());
              } else if (_ptype == fLocal) {
                _Raddam_ADCvsEvn.fill((int)e.eventAuxiliary().id().event(), digi[i].adc());
              }
            }
          }
        }
      }
      continue;
    }
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

    // Below we are requiring both laser type equals 24 and uHTR event type from crate:slot 22:01 equals 14 to confirm this is a HFRaddam laser signal
    //  laser type check
    //uint32_t laserType = cumn->valueUserWord(0);
    //if (laserType != _laserType)
    //  return false;

    // uHTR event type check from crate:slot 22:01 for HF Raddam
    bool eventflag_uHTR = false;
    edm::Handle<FEDRawDataCollection> craw;
    if (!e.getByToken(_tokFEDs, craw))
      _logger.dqmthrow("Collection FEDRawDataCollection isn't available " + _tagFEDs.label() + " " +
                       _tagFEDs.instance());

    for (int fed = FEDNumbering::MINHCALFEDID; fed <= FEDNumbering::MAXHCALuTCAFEDID && !eventflag_uHTR; fed++) {
      if ((fed > FEDNumbering::MAXHCALFEDID && fed < FEDNumbering::MINHCALuTCAFEDID) ||
          fed > FEDNumbering::MAXHCALuTCAFEDID)
        continue;
      FEDRawData const& raw = craw->FEDData(fed);
      if (raw.size() < constants::RAW_EMPTY)
        continue;

      hcal::AMC13Header const* hamc13 = (hcal::AMC13Header const*)raw.data();
      if (!hamc13)
        continue;

      for (int iamc = 0; iamc < hamc13->NAMC(); iamc++) {
        HcalUHTRData uhtr(hamc13->AMCPayload(iamc), hamc13->AMCSize(iamc));
        if (static_cast<int>(uhtr.crateId()) == 22 && static_cast<int>(uhtr.slot()) == 1)
          if (uhtr.getEventType() == constants::EVENTTYPE_HFRADDAM) {
            eventflag_uHTR = true;
            break;
          }
      }
    }
    if (eventflag_uHTR)
      return true;

  } else if (_ptype == fLocal) {
    //	local, just return true as all the settings will be done in cfg
    return true;
  }

  return false;
}

DEFINE_FWK_MODULE(HFRaddamTask);
