#include "DQM/HcalCommon/interface/DQTask.h"

#include "FWCore/Utilities/interface/ESInputTag.h"
#include "FWCore/Utilities/interface/Transition.h"

namespace hcaldqm {
  using namespace constants;
  DQTask::DQTask(edm::ParameterSet const &ps)
      : DQModule(ps),
        _cEvsTotal(_name, "EventsTotal"),
        _cEvsPerLS(_name, "EventsPerLS"),
        _cRunKeyVal(_name, "RunKeyValue"),
        _cRunKeyName(_name, "RunKeyName"),
        _cProcessingTypeName(_name, "ProcessingType"),
        _procLSs(0),
        hcalDbServiceToken_(esConsumes<HcalDbService, HcalDbRecord, edm::Transition::BeginRun>()),
        runInfoToken_(esConsumes<RunInfo, RunInfoRcd, edm::Transition::BeginRun>()),
        hcalChannelQualityToken_(
            esConsumes<HcalChannelQuality, HcalChannelQualityRcd, edm::Transition::BeginLuminosityBlock>(
                edm::ESInputTag("", "withTopo"))) {
    //	tags and Tokens
    _tagRaw = ps.getUntrackedParameter<edm::InputTag>("tagRaw", edm::InputTag("rawDataCollector"));
    _tokRaw = consumes<FEDRawDataCollection>(_tagRaw);
  }

  /*
 *	By design, all the sources will ahve this function inherited and will
 *	never override.
 */
  void DQTask::analyze(edm::Event const &e, edm::EventSetup const &es) {
    this->_resetMonitors(fEvent);
    _logger.debug(_name + " processing");
    if (!this->_isApplicable(e))
      return;

    _evsTotal++;
    _cEvsTotal.fill(_evsTotal);

    auto lumiCache = luminosityBlockCache(e.getLuminosityBlock().index());
    lumiCache->EvtCntLS++;
    _evsPerLS = lumiCache->EvtCntLS;
    _cEvsPerLS.fill(_evsPerLS);

    this->_process(e, es);
  }

  void DQTask::bookHistograms(DQMStore::IBooker &ib, edm::Run const &r, edm::EventSetup const &es) {
    //	initialize some containers to be used by all modules
    _xQuality.initialize(hashfunctions::fDChannel);

    //	get the run info FEDs - FEDs registered at cDAQ
    //	and determine if there are any HCAL FEDs in.
    //	push them as ElectronicsIds into the vector
    if (auto runInfoRec = es.tryToGet<RunInfoRcd>()) {
      const RunInfo &runInfo = es.getData(runInfoToken_);
      std::vector<int> vfeds = runInfo.m_fed_in;
      for (std::vector<int>::const_iterator it = vfeds.begin(); it != vfeds.end(); ++it) {
        if (*it >= constants::FED_VME_MIN && *it <= FED_VME_MAX)
          _vcdaqEids.push_back(
              HcalElectronicsId(constants::FIBERCH_MIN, constants::FIBER_VME_MIN, SPIGOT_MIN, (*it) - FED_VME_MIN)
                  .rawId());
        else if (*it >= constants::FED_uTCA_MIN && *it <= FEDNumbering::MAXHCALuTCAFEDID) {
          std::pair<uint16_t, uint16_t> cspair = utilities::fed2crate(*it);
          _vcdaqEids.push_back(
              HcalElectronicsId(cspair.first, cspair.second, FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
        }
      }
    }

    //	book some base guys
    _cEvsTotal.book(ib, _subsystem);
    _cEvsPerLS.book(ib, _subsystem);
    _cRunKeyVal.book(ib, _subsystem);
    _cRunKeyName.book(ib, _subsystem);
    _cProcessingTypeName.book(ib, _subsystem);

    //	fill what you can now
    _cRunKeyVal.fill(_runkeyVal);
    _cRunKeyName.fill(_runkeyName);
    _cProcessingTypeName.fill(pTypeNames[_ptype]);

    // Load conditions and emap
    _dbService = es.getHandle(hcalDbServiceToken_);
    _emap = _dbService->getHcalMapping();
  }

  void DQTask::dqmBeginRun(edm::Run const &, edm::EventSetup const &) {
    this->_resetMonitors(fEvent);
    this->_resetMonitors(f1LS);
    this->_resetMonitors(f10LS);
    this->_resetMonitors(f50LS);
    this->_resetMonitors(f100LS);
  }

  std::shared_ptr<hcaldqm::Cache> DQTask::globalBeginLuminosityBlock(edm::LuminosityBlock const &lb,
                                                                     edm::EventSetup const &es) const {
    auto d = std::make_shared<hcaldqm::Cache>();
    d->currentLS = lb.luminosityBlock();
    d->EvtCntLS = 0;

    /*   //// these resets were not useful anymore
    this->_resetMonitors(f1LS);
    if (_procLSs % 10 == 0)
      this->_resetMonitors(f10LS);
    if (_procLSs % 50 == 0)
      this->_resetMonitors(f50LS);
    if (_procLSs % 100 == 0)
      this->_resetMonitors(f100LS);
*/

    //	get the Channel Quality Status for all the channels
    d->xQuality.initialize(hashfunctions::fDChannel);
    d->xQuality.reset();
    const HcalChannelQuality &cq = es.getData(hcalChannelQualityToken_);
    std::vector<DetId> detids = cq.getAllChannels();
    for (std::vector<DetId>::const_iterator it = detids.begin(); it != detids.end(); ++it) {
      //	if unknown skip
      if (HcalGenericDetId(*it).genericSubdet() == HcalGenericDetId::HcalGenUnknown)
        continue;

      if (HcalGenericDetId(*it).isHcalDetId()) {
        HcalDetId did(*it);
        uint32_t mask = (cq.getValues(did))->getValue();
        if (mask != 0) {
          d->xQuality.push(did, mask);
        }
      }
    }
    return d;
  }

  void DQTask::globalEndLuminosityBlock(edm::LuminosityBlock const &, edm::EventSetup const &) { _procLSs++; }

  void DQTask::_resetMonitors(UpdateFreq uf) {
    //	reset per event
    switch (uf) {
      case fEvent:
        break;
      case f1LS:
        _evsPerLS = 0;
        break;
      case f10LS:
        break;
      case f50LS:
        break;
      case f100LS:
        break;
      default:
        break;
    }
  }

  int DQTask::_getCalibType(edm::Event const &e) {
    int calibType = 0;

    edm::Handle<FEDRawDataCollection> craw;
    if (!e.getByToken(_tokRaw, craw))
      _logger.dqmthrow("Collection FEDRawDataCollection isn't available " + _tagRaw.label() + " " + _tagRaw.instance());

    int badFEDs = 0;
    std::vector<int> types(8, 0);
    for (int i = FED_VME_MIN; i <= FED_VME_MAX; i++) {
      FEDRawData const &fd = craw->FEDData(i);
      if (fd.size() < 24) {
        badFEDs++;
        continue;
      }
      int cval = (int)((HcalDCCHeader const *)(fd.data()))->getCalibType();
      if (cval > 7)
        _logger.warn("Unexpected Calib Type in FED " + std::to_string(i));
      types[cval]++;
    }
    for (int i = FED_uTCA_MIN; i <= FED_uTCA_MAX; i++) {
      FEDRawData const &fd = craw->FEDData(i);
      if (fd.size() < 24) {
        badFEDs++;
        continue;
      }
      int cval = (int)((HcalDCCHeader const *)(fd.data()))->getCalibType();
      if (cval > 7)
        _logger.warn("Unexpected Calib Type in FED " + std::to_string(i));
      types[cval]++;
    }

    int max = 0;
    for (unsigned int ic = 0; ic < 8; ic++) {
      if (types[ic] > max) {
        max = types[ic];
        calibType = ic;
      }
    }
    if (max != (FED_VME_NUM + (FED_uTCA_MAX - FED_uTCA_MIN + 1) - badFEDs))
      _logger.warn("Conflicting Calibration Types found. Assigning " + std::to_string(calibType));

    return calibType;
  }
}  // namespace hcaldqm
