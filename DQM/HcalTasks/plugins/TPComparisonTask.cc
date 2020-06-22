
#include "DQM/HcalTasks/interface/TPComparisonTask.h"

using namespace hcaldqm;
using namespace hcaldqm::constants;
TPComparisonTask::TPComparisonTask(edm::ParameterSet const& ps) : DQTask(ps) {
  //	tags and tokens
  _tag1 = ps.getUntrackedParameter<edm::InputTag>("tag1", edm::InputTag("hcalDigis"));
  _tag2 = ps.getUntrackedParameter<edm::InputTag>("tag2", edm::InputTag("vmeDigis"));
  _tok1 = consumes<HcalTrigPrimDigiCollection>(_tag1);
  _tok2 = consumes<HcalTrigPrimDigiCollection>(_tag2);

  //	tmp flags
  _skip1x1 = ps.getUntrackedParameter<bool>("skip1x1", true);
}

/* virtual */ void TPComparisonTask::bookHistograms(DQMStore::IBooker& ib,
                                                    edm::Run const& r,
                                                    edm::EventSetup const& es) {
  DQTask::bookHistograms(ib, r, es);

  //	GET WHAT YOU NEED
  edm::ESHandle<HcalDbService> dbs;
  es.get<HcalDbRecord>().get(dbs);
  edm::ESHandle<HcalElectronicsMap> item;
  es.get<HcalElectronicsMapRcd>().get("full", item);
  _emap = item.product();
  if (_ptype != fOffline) {  // hidefed2crate
    std::vector<int> vFEDs = hcaldqm::utilities::getFEDList(_emap);
    std::vector<int> vFEDsVME = hcaldqm::utilities::getFEDVMEList(_emap);
    std::vector<int> vFEDsuTCA = hcaldqm::utilities::getFEDuTCAList(_emap);
  }
  std::vector<uint32_t> vhashVME;
  std::vector<uint32_t> vhashuTCA;
  vhashVME.push_back(
      HcalElectronicsId(constants::FIBERCH_MIN, constants::FIBER_VME_MIN, SPIGOT_MIN, CRATE_VME_MIN).rawId());
  vhashuTCA.push_back(HcalElectronicsId(CRATE_uTCA_MIN, SLOT_uTCA_MIN, FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
  _filter_VME.initialize(filter::fFilter, hcaldqm::hashfunctions::fElectronics, vhashVME);
  _filter_uTCA.initialize(filter::fFilter, hcaldqm::hashfunctions::fElectronics, vhashuTCA);

  //	INTIALIZE CONTAINERS
  for (unsigned int i = 0; i < 4; i++) {
    _cEt_TTSubdet[i].initialize(_name,
                                "Et",
                                hcaldqm::hashfunctions::fTTSubdet,
                                new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fEtCorr_256),
                                new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fEtCorr_256),
                                new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),
                                0);
    _cFG_TTSubdet[i].initialize(_name,
                                "FG",
                                hcaldqm::hashfunctions::fTTSubdet,
                                new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fFG),
                                new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fFG),
                                new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),
                                0);
  }
  _cEtall_TTSubdet.initialize(_name,
                              "Et",
                              hcaldqm::hashfunctions::fTTSubdet,
                              new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fEtCorr_256),
                              new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fEtCorr_256),
                              new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),
                              0);

  if (_ptype != fOffline) {  // hidefed2crate
    _cMsn_FEDVME.initialize(_name,
                            "Missing",
                            hcaldqm::hashfunctions::fFED,
                            new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSpigot),
                            new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSLBSLBCh),
                            new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                            0);
    _cMsn_FEDuTCA.initialize(_name,
                             "Missing",
                             hcaldqm::hashfunctions::fFED,
                             new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSlotuTCA),
                             new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fFiberuTCATPFiberChuTCATP),
                             new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                             0);
    _cEtMsm_FEDVME.initialize(_name,
                              "EtMsm",
                              hcaldqm::hashfunctions::fFED,
                              new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSpigot),
                              new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSLBSLBCh),
                              new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                              0);
    _cEtMsm_FEDuTCA.initialize(_name,
                               "EtMsm",
                               hcaldqm::hashfunctions::fFED,
                               new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSlotuTCA),
                               new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fFiberuTCATPFiberChuTCATP),
                               new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                               0);
    _cFGMsm_FEDVME.initialize(_name,
                              "FGMsm",
                              hcaldqm::hashfunctions::fFED,
                              new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSpigot),
                              new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSLBSLBCh),
                              new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                              0);
    _cFGMsm_FEDuTCA.initialize(_name,
                               "FGMsm",
                               hcaldqm::hashfunctions::fFED,
                               new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSlotuTCA),
                               new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fFiberuTCATPFiberChuTCATP),
                               new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                               0);
  }

  _cMsnuTCA.initialize(_name,
                       "Missing",
                       new hcaldqm::quantity::TrigTowerQuantity(hcaldqm::quantity::fTTieta),
                       new hcaldqm::quantity::TrigTowerQuantity(hcaldqm::quantity::fTTiphi),
                       new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                       0);
  _cMsnVME.initialize(_name,
                      "Missing",
                      new hcaldqm::quantity::TrigTowerQuantity(hcaldqm::quantity::fTTieta),
                      new hcaldqm::quantity::TrigTowerQuantity(hcaldqm::quantity::fTTiphi),
                      new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                      0);
  _cEtMsm.initialize(_name,
                     "EtMsm",
                     new hcaldqm::quantity::TrigTowerQuantity(hcaldqm::quantity::fTTieta),
                     new hcaldqm::quantity::TrigTowerQuantity(hcaldqm::quantity::fTTiphi),
                     new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                     0);
  _cFGMsm.initialize(_name,
                     "FGMsm",
                     new hcaldqm::quantity::TrigTowerQuantity(hcaldqm::quantity::fTTieta),
                     new hcaldqm::quantity::TrigTowerQuantity(hcaldqm::quantity::fTTiphi),
                     new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                     0);

  char aux[20];
  for (unsigned int i = 0; i < 4; i++) {
    sprintf(aux, "TS%d", i);
    _cEt_TTSubdet[i].book(ib, _emap, _subsystem, aux);
    _cFG_TTSubdet[i].book(ib, _emap, _subsystem, aux);
  }
  _cEtall_TTSubdet.book(ib, _emap, _subsystem);
  if (_ptype != fOffline) {  // hidefed2crate
    _cMsn_FEDVME.book(ib, _emap, _filter_uTCA, _subsystem);
    _cEtMsm_FEDVME.book(ib, _emap, _filter_uTCA, _subsystem);
    _cFGMsm_FEDVME.book(ib, _emap, _filter_uTCA, _subsystem);
    _cMsn_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
    _cEtMsm_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
    _cFGMsm_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
  }

  _cMsnuTCA.book(ib, _subsystem, std::string("uTCA"));
  _cMsnVME.book(ib, _subsystem, std::string("VME"));
  _cEtMsm.book(ib, _subsystem);
  _cFGMsm.book(ib, _subsystem);

  _ehashmapuTCA.initialize(_emap, hcaldqm::electronicsmap::fT2EHashMap, _filter_VME);
  _ehashmapVME.initialize(_emap, hcaldqm::electronicsmap::fT2EHashMap, _filter_uTCA);
  //	_ehashmap.print();
  //	_cMsn_depth.book(ib);
  //	_cEtMsm_depth.book(ib);
  //	_cFGMsm_depth.book(ib);
}

/* virtual */ void TPComparisonTask::_resetMonitors(hcaldqm::UpdateFreq uf) { DQTask::_resetMonitors(uf); }

/* virtual */ void TPComparisonTask::_process(edm::Event const& e, edm::EventSetup const& es) {
  edm::Handle<HcalTrigPrimDigiCollection> coll1;
  edm::Handle<HcalTrigPrimDigiCollection> coll2;

  if (!e.getByToken(_tok1, coll1))
    _logger.dqmthrow("Collection HcalTrigPrimDigiCollection isn't available" + _tag1.label() + " " + _tag1.instance());
  if (!e.getByToken(_tok2, coll2))
    _logger.dqmthrow("Collection HcalTrigPrimDigiCollection isn't available" + _tag2.label() + " " + _tag2.instance());

  //	assume always coll1 is primary (uTCA) and coll2 is secondary(VME)
  for (HcalTrigPrimDigiCollection::const_iterator it1 = coll1->begin(); it1 != coll1->end(); ++it1) {
    //	iterate thru utca collection
    //	get the same detid digi from vme collection
    //	if missing - fill vme missing
    //	else correlate
    //	tmp
    if (_skip1x1)
      if (it1->id().version() > 0)
        continue;
    //	\tmp

    HcalTrigTowerDetId tid = it1->id();
    HcalTrigPrimDigiCollection::const_iterator it2 = coll2->find(HcalTrigTowerDetId(tid.ieta(), tid.iphi(), 0));
    HcalElectronicsId eid1 = HcalElectronicsId(_ehashmapuTCA.lookup(tid));
    HcalElectronicsId eid2 = HcalElectronicsId(_ehashmapVME.lookup(tid));

    if (it2 == coll2->end()) {
      //	missing from VME collection
      _cMsnVME.fill(tid);
      if (_ptype != fOffline) {  // hidefed2crate
        _cMsn_FEDVME.fill(eid2);
      }
      for (int i = 0; i < it1->size(); i++) {
        _cEtall_TTSubdet.fill(tid, it1->sample(i).compressedEt(), -2);
        _cEt_TTSubdet[i].fill(tid, it1->sample(i).compressedEt(), -2);
      }
    } else
      for (int i = 0; i < it1->size(); i++) {
        _cEtall_TTSubdet.fill(tid, it1->sample(i).compressedEt(), it2->sample(i).compressedEt());
        _cEt_TTSubdet[i].fill(tid, it1->sample(i).compressedEt(), it2->sample(i).compressedEt());
        _cFG_TTSubdet[i].fill(tid, it1->sample(i).fineGrain(), it2->sample(i).fineGrain());
        if (it1->sample(i).compressedEt() != it2->sample(i).compressedEt()) {
          if (_ptype != fOffline) {  // hidefed2crate
            _cEtMsm_FEDuTCA.fill(eid1);
            _cEtMsm_FEDVME.fill(eid2);
          }
          _cEtMsm.fill(tid);
        }
        if (it1->sample(i).fineGrain() != it2->sample(i).fineGrain()) {
          if (_ptype != fOffline) {  // hidefed2crate
            _cFGMsm_FEDuTCA.fill(eid1);
            _cFGMsm_FEDVME.fill(eid2);
          }
          _cFGMsm.fill(tid);
        }
      }
  }
  for (HcalTrigPrimDigiCollection::const_iterator it2 = coll2->begin(); it2 != coll2->end(); ++it2) {
    //	itearte thru VME
    //	find utca tp digi by detid
    //	check if present of missing
    HcalTrigTowerDetId tid = it2->id();
    if (_skip1x1)
      if (tid.version() > 0)
        continue;

    HcalTrigPrimDigiCollection::const_iterator it1 = coll1->find(HcalTrigTowerDetId(tid.ieta(), tid.iphi(), 0));
    if (it1 == coll1->end()) {
      HcalElectronicsId eid1 = HcalElectronicsId(_ehashmapuTCA.lookup(tid));
      if (_ptype != fOffline) {  // hidefed2crate
        _cMsn_FEDuTCA.fill(eid1);
      }
      _cMsnuTCA.fill(tid);
      for (int i = 0; i < it2->size(); i++) {
        _cEtall_TTSubdet.fill(tid, -2, it2->sample(i).compressedEt());
        _cEt_TTSubdet[i].fill(tid, -2, it2->sample(i).compressedEt());
      }
    }
  }
}

/* virtual */ void TPComparisonTask::dqmEndLuminosityBlock(edm::LuminosityBlock const& lb, edm::EventSetup const& es) {
  //	in the end always
  DQTask::dqmEndLuminosityBlock(lb, es);
}

DEFINE_FWK_MODULE(TPComparisonTask);
