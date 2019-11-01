
#include "DQM/HcalTasks/interface/DigiComparisonTask.h"

using namespace hcaldqm;
using namespace hcaldqm::constants;

DigiComparisonTask::DigiComparisonTask(edm::ParameterSet const& ps) : DQTask(ps) {
  //	tags and tokens
  _tagHBHE1 = ps.getUntrackedParameter<edm::InputTag>("tagHBHE1", edm::InputTag("hcalDigis"));
  _tagHBHE2 = ps.getUntrackedParameter<edm::InputTag>("tagHBHE2", edm::InputTag("vmeDigis"));
  _tokHBHE1 = consumes<HBHEDigiCollection>(_tagHBHE1);
  _tokHBHE2 = consumes<HBHEDigiCollection>(_tagHBHE2);
}
/* virtual */ void DigiComparisonTask::bookHistograms(DQMStore::IBooker& ib,
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
    std::vector<int> vFEDs = utilities::getFEDList(_emap);
    std::vector<int> vFEDsVME = utilities::getFEDVMEList(_emap);
    std::vector<int> vFEDsuTCA = utilities::getFEDuTCAList(_emap);
  }
  std::vector<uint32_t> vhashVME;
  std::vector<uint32_t> vhashuTCA;
  vhashVME.push_back(
      HcalElectronicsId(constants::FIBERCH_MIN, constants::FIBER_VME_MIN, SPIGOT_MIN, CRATE_VME_MIN).rawId());
  vhashuTCA.push_back(HcalElectronicsId(CRATE_uTCA_MIN, SLOT_uTCA_MIN, FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
  _filter_VME.initialize(filter::fFilter, hashfunctions::fElectronics, vhashVME);
  _filter_uTCA.initialize(filter::fFilter, hashfunctions::fElectronics, vhashuTCA);

  //	INITIALIZE
  for (unsigned int i = 0; i < 10; i++) {
    _cADC_Subdet[i].initialize(_name,
                               "ADC",
                               hashfunctions::fSubdet,
                               new quantity::ValueQuantity(quantity::fADCCorr_128),
                               new quantity::ValueQuantity(quantity::fADCCorr_128),
                               new quantity::ValueQuantity(quantity::fN, true),
                               0);
  }
  _cADCall_Subdet.initialize(_name,
                             "ADC",
                             hashfunctions::fSubdet,
                             new quantity::ValueQuantity(quantity::fADCCorr_128),
                             new quantity::ValueQuantity(quantity::fADCCorr_128),
                             new quantity::ValueQuantity(quantity::fN, true),
                             0);
  _cADCMsnuTCA_Subdet.initialize(_name,
                                 "ADCMsnuTCA",
                                 hashfunctions::fSubdet,
                                 new quantity::ValueQuantity(quantity::fADC_128),
                                 new quantity::ValueQuantity(quantity::fN, true),
                                 0);
  _cADCMsnVME_Subdet.initialize(_name,
                                "ADCMsnVME",
                                hashfunctions::fSubdet,
                                new quantity::ValueQuantity(quantity::fADC_128),
                                new quantity::ValueQuantity(quantity::fN, true),
                                0);
  _cMsm_depth.initialize(_name,
                         "Mismatched",
                         hashfunctions::fdepth,
                         new quantity::DetectorQuantity(quantity::fieta),
                         new quantity::DetectorQuantity(quantity::fiphi),
                         new quantity::ValueQuantity(quantity::fN),
                         0);
  if (_ptype != fOffline) {  // hidefed2crate
    _cMsm_FEDVME.initialize(_name,
                            "Mismatched",
                            hashfunctions::fFED,
                            new quantity::ElectronicsQuantity(quantity::fSpigot),
                            new quantity::ElectronicsQuantity(quantity::fFiberVMEFiberCh),
                            new quantity::ValueQuantity(quantity::fN),
                            0);
    _cMsm_FEDuTCA.initialize(_name,
                             "Mismatched",
                             hashfunctions::fFED,
                             new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
                             new quantity::ElectronicsQuantity(quantity::fFiberuTCAFiberCh),
                             new quantity::ValueQuantity(quantity::fN),
                             0);
  }
  _cMsnVME_depth.initialize(_name,
                            "Missing",
                            hashfunctions::fdepth,
                            new quantity::DetectorQuantity(quantity::fieta),
                            new quantity::DetectorQuantity(quantity::fiphi),
                            new quantity::ValueQuantity(quantity::fN),
                            0);
  _cMsnuTCA_depth.initialize(_name,
                             "Missing",
                             hashfunctions::fdepth,
                             new quantity::DetectorQuantity(quantity::fieta),
                             new quantity::DetectorQuantity(quantity::fiphi),
                             new quantity::ValueQuantity(quantity::fN),
                             0);
  if (_ptype != fOffline) {  // hidefed2crate
    _cMsn_FEDVME.initialize(_name,
                            "Missing",
                            hashfunctions::fFED,
                            new quantity::ElectronicsQuantity(quantity::fSpigot),
                            new quantity::ElectronicsQuantity(quantity::fFiberVMEFiberCh),
                            new quantity::ValueQuantity(quantity::fN),
                            0);
    _cMsn_FEDuTCA.initialize(_name,
                             "Missing",
                             hashfunctions::fFED,
                             new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
                             new quantity::ElectronicsQuantity(quantity::fFiberuTCAFiberCh),
                             new quantity::ValueQuantity(quantity::fN),
                             0);
  }

  //	BOOK
  char aux[20];
  for (unsigned int i = 0; i < 10; i++) {
    sprintf(aux, "TS%d", i);
    _cADC_Subdet[i].book(ib, _emap, _subsystem, aux);
  }
  _cADCall_Subdet.book(ib, _emap, _subsystem);
  _cADCMsnVME_Subdet.book(ib, _emap, _subsystem);
  _cADCMsnuTCA_Subdet.book(ib, _emap, _subsystem);
  _cMsm_depth.book(ib, _emap, _subsystem);
  _cMsnVME_depth.book(ib, _emap, _subsystem, std::string("VME"));
  _cMsnuTCA_depth.book(ib, _emap, _subsystem, std::string("uTCA"));
  if (_ptype != fOffline) {  // hidefed2crate
    _cMsm_FEDVME.book(ib, _emap, _filter_uTCA, _subsystem);
    _cMsn_FEDVME.book(ib, _emap, _filter_uTCA, _subsystem);
    _cMsm_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
    _cMsn_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
  }

  _ehashmapuTCA.initialize(_emap, hcaldqm::electronicsmap::fD2EHashMap, _filter_VME);
  _ehashmapVME.initialize(_emap, hcaldqm::electronicsmap::fD2EHashMap, _filter_uTCA);
}

/* virtual */ void DigiComparisonTask::_resetMonitors(hcaldqm::UpdateFreq uf) { DQTask::_resetMonitors(uf); }

/* virtual */ void DigiComparisonTask::_process(edm::Event const& e, edm::EventSetup const& es) {
  edm::Handle<HBHEDigiCollection> chbhe1;
  edm::Handle<HBHEDigiCollection> chbhe2;

  if (!e.getByToken(_tokHBHE1, chbhe1))
    _logger.dqmthrow("Collection HBHEDigiCollection isn't available" + _tagHBHE1.label() + " " + _tagHBHE1.instance());
  if (!e.getByToken(_tokHBHE2, chbhe2))
    _logger.dqmthrow("Collection HBHEDigiCollection isn't available" + _tagHBHE2.label() + " " + _tagHBHE2.instance());

  //	assume that coll1 is primary(uTCA) and coll2 is secondary(VME)
  //	uTCA is X and VME is Y axis
  for (HBHEDigiCollection::const_iterator it1 = chbhe1->begin(); it1 != chbhe1->end(); ++it1) {
    //	iterate thru the utca collection
    //	get the same detid digi from vme collection
    //	if missing - fill vme missing
    //	else correlate
    HcalDetId did = it1->id();
    HcalElectronicsId eid1 = it1->elecId();
    HBHEDigiCollection::const_iterator it2 = chbhe2->find(did);

    //	get the eid for vme by did
    HcalElectronicsId eid2 = HcalElectronicsId(_ehashmapVME.lookup(did));
    if (it2 == chbhe2->end()) {
      //	fill the depth plot
      _cMsnVME_depth.fill(did);
      if (_ptype != fOffline) {  // hidefed2crate
        _cMsn_FEDVME.fill(eid2);
      }
      for (int i = 0; i < it1->size(); i++) {
        _cADCMsnVME_Subdet.fill(did, it1->sample(i).adc());
        _cADCall_Subdet.fill(did, it1->sample(i).adc(), -2);
        _cADC_Subdet[i].fill(did, it1->sample(i).adc(), -2);
      }
    } else
      for (int i = 0; i < it1->size(); i++) {
        _cADCall_Subdet.fill(did, double(it1->sample(i).adc()), double(it2->sample(i).adc()));
        _cADC_Subdet[i].fill(did, double(it1->sample(i).adc()), double(it2->sample(i).adc()));
        if (it1->sample(i).adc() != it2->sample(i).adc()) {
          //	fill depth, uTCA and VME as well for which guys
          //	mismatches happen
          _cMsm_depth.fill(did);
          if (_ptype != fOffline) {  // hidefed2crate
            _cMsm_FEDVME.fill(eid2);
            _cMsm_FEDuTCA.fill(eid1);
          }
        }
      }
  }
  for (HBHEDigiCollection::const_iterator it2 = chbhe2->begin(); it2 != chbhe2->end(); ++it2) {
    //	itearte thru VME
    //	find utca digi by detid
    //	check if present or missing
    HcalDetId did = it2->id();
    HBHEDigiCollection::const_iterator it1 = chbhe1->find(did);
    if (it1 == chbhe1->end()) {
      HcalElectronicsId eid1 = HcalElectronicsId(_ehashmapuTCA.lookup(did));
      if (_ptype != fOffline) {  // hidefed2crate
        _cMsn_FEDuTCA.fill(eid1);
      }
      for (int i = 0; i < it2->size(); i++) {
        _cADCMsnuTCA_Subdet.fill(did, it2->sample(i).adc());
        _cADCall_Subdet.fill(did, -2, it2->sample(i).adc());
        _cADC_Subdet[i].fill(did, -2, it2->sample(i).adc());
      }
    }
  }
}

/* virtual */ void DigiComparisonTask::dqmEndLuminosityBlock(edm::LuminosityBlock const& lb,
                                                             edm::EventSetup const& es) {
  //	in the end always
  DQTask::dqmEndLuminosityBlock(lb, es);
}

DEFINE_FWK_MODULE(DigiComparisonTask);
