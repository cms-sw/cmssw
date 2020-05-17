#include "DQM/HcalCommon/interface/DQHarvester.h"
#include "FWCore/Framework/interface/Run.h"

namespace hcaldqm {
  using namespace constants;

  DQHarvester::DQHarvester(edm::ParameterSet const &ps) : DQModule(ps) {}

  void DQHarvester::beginRun(edm::Run const &r, edm::EventSetup const &es) {
    if (_ptype == fLocal)
      if (r.runAuxiliary().run() == 1)
        return;

    //      TEMPORARY FIX
    if (_ptype != fOffline) {  // hidefed2crate
      _vhashFEDs.clear();
      _vcdaqEids.clear();
    }

    //	- get the Hcal Electronics Map
    //	- collect all the FED numbers and FED's rawIds
    edm::ESHandle<HcalDbService> dbs;
    es.get<HcalDbRecord>().get(dbs);
    _emap = dbs->getHcalMapping();

    if (_ptype != fOffline) {  // hidefed2crate
      _vFEDs = utilities::getFEDList(_emap);
      for (int _vFED : _vFEDs) {
        //
        //	FIXME
        //	until there exists a map of FED2Crate and Crate2FED,
        //	all the unknown Crates will be mapped to 0...
        //
        if (_vFED == 0) {
          _vhashFEDs.push_back(HcalElectronicsId(0, SLOT_uTCA_MIN, FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
          continue;
        }

        if (_vFED > FED_VME_MAX) {
          std::pair<uint16_t, uint16_t> cspair = utilities::fed2crate(_vFED);
          _vhashFEDs.push_back(
              HcalElectronicsId(cspair.first, cspair.second, FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
        } else
          _vhashFEDs.push_back(HcalElectronicsId(FIBERCH_MIN, FIBER_VME_MIN, SPIGOT_MIN, _vFED - FED_VME_MIN).rawId());
      }

      //	get the FEDs registered at cDAQ
      if (auto runInfoRec = es.tryToGet<RunInfoRcd>()) {
        edm::ESHandle<RunInfo> ri;
        runInfoRec->get(ri);
        std::vector<int> vfeds = ri->m_fed_in;
        for (int vfed : vfeds) {
          if (vfed >= constants::FED_VME_MIN && vfed <= FED_VME_MAX)
            _vcdaqEids.push_back(
                HcalElectronicsId(constants::FIBERCH_MIN, constants::FIBER_VME_MIN, SPIGOT_MIN, vfed - FED_VME_MIN)
                    .rawId());
          else if (vfed >= constants::FED_uTCA_MIN && vfed <= FEDNumbering::MAXHCALuTCAFEDID) {
            std::pair<uint16_t, uint16_t> cspair = utilities::fed2crate(vfed);
            _vcdaqEids.push_back(
                HcalElectronicsId(cspair.first, cspair.second, FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
          }
        }
      }
    } else {
      _vCrates = utilities::getCrateList(_emap);
      std::map<int, uint32_t> crateHashMap = utilities::getCrateHashMap(_emap);
      for (auto &it_crate : _vCrates) {
        _vhashCrates.push_back(crateHashMap[it_crate]);
      }
    }

    // Initialize channel quality masks, but do not load (changed for 10_4_X,
    // moving to LS granularity)
    _xQuality.initialize(hashfunctions::fDChannel);
  }

  void DQHarvester::dqmBeginLuminosityBlock(DQMStore::IBooker &ib,
                                            DQMStore::IGetter &ig,
                                            edm::LuminosityBlock const &lb,
                                            edm::EventSetup const &es) {
    //	get the Hcal Channels Quality for channels that are not 0
    edm::ESHandle<HcalChannelQuality> hcq;
    es.get<HcalChannelQualityRcd>().get("withTopo", hcq);
    const HcalChannelQuality *cq = hcq.product();
    std::vector<DetId> detids = cq->getAllChannels();
    for (auto detid : detids) {
      if (HcalGenericDetId(detid).genericSubdet() == HcalGenericDetId::HcalGenUnknown)
        continue;

      if (HcalGenericDetId(detid).isHcalDetId()) {
        HcalDetId did(detid);
        uint32_t mask = (cq->getValues(did))->getValue();
        if (mask != 0)
          _xQuality.push(did, mask);
      }
    }
  }

  void DQHarvester::dqmEndLuminosityBlock(DQMStore::IBooker &ib,
                                          DQMStore::IGetter &ig,
                                          edm::LuminosityBlock const &lb,
                                          edm::EventSetup const &es) {
    //	should be the same - just in case!
    _currentLS = lb.luminosityBlock();
    _totalLS++;
    _dqmEndLuminosityBlock(ib, ig, lb, es);
  }
  void DQHarvester::dqmEndJob(DQMStore::IBooker &ib, DQMStore::IGetter &ig) { _dqmEndJob(ib, ig); }
}  // namespace hcaldqm
