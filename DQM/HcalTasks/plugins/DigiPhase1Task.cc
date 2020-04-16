#include "DQM/HcalTasks/interface/DigiPhase1Task.h"

using namespace hcaldqm;
using namespace hcaldqm::constants;
using namespace hcaldqm::filter;

DigiPhase1Task::DigiPhase1Task(edm::ParameterSet const& ps) : DQTask(ps) {
  _tagHBHE = ps.getUntrackedParameter<edm::InputTag>("tagHBHE", edm::InputTag("hcalDigis"));
  _tagHO = ps.getUntrackedParameter<edm::InputTag>("tagHO", edm::InputTag("hcalDigis"));
  _tagHF = ps.getUntrackedParameter<edm::InputTag>("tagHF", edm::InputTag("hcalDigis"));

  _tokHBHE = consumes<QIE11DigiCollection>(_tagHBHE);
  _tokHO = consumes<HODigiCollection>(_tagHO);
  _tokHF = consumes<QIE10DigiCollection>(_tagHF);

  _cutSumQ_HBHE = ps.getUntrackedParameter<double>("cutSumQ_HBHE", 20);
  _cutSumQ_HO = ps.getUntrackedParameter<double>("cutSumQ_HO", 20);
  _cutSumQ_HF = ps.getUntrackedParameter<double>("cutSumQ_HF", 20);
  _thresh_unihf = ps.getUntrackedParameter<double>("thresh_unihf", 0.2);

  _vflags.resize(nDigiFlag);
  _vflags[fUni] = hcaldqm::flag::Flag("UniSlotHF");
  _vflags[fDigiSize] = hcaldqm::flag::Flag("DigiSize");
  _vflags[fNChsHF] = hcaldqm::flag::Flag("NChsHF");
  _vflags[fUnknownIds] = hcaldqm::flag::Flag("UnknownIds");
}

/* virtual */ void DigiPhase1Task::bookHistograms(DQMStore::IBooker& ib, edm::Run const& r, edm::EventSetup const& es) {
  DQTask::bookHistograms(ib, r, es);

  //  GET WHAT YOU NEED
  std::vector<uint32_t> vVME;
  std::vector<uint32_t> vuTCA;
  vVME.push_back(
      HcalElectronicsId(constants::FIBERCH_MIN, constants::FIBER_VME_MIN, SPIGOT_MIN, CRATE_VME_MIN).rawId());
  vuTCA.push_back(HcalElectronicsId(CRATE_uTCA_MIN, SLOT_uTCA_MIN, FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
  _filter_VME.initialize(filter::fFilter, hcaldqm::hashfunctions::fElectronics, vVME);
  _filter_uTCA.initialize(filter::fFilter, hcaldqm::hashfunctions::fElectronics, vuTCA);

  //  INITIALIZE FIRST
  _cADC_SubdetPM.initialize(_name,
                            "ADC",
                            hcaldqm::hashfunctions::fSubdetPM,
                            new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fADC_128),
                            new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),
                            0);
  _cfC_SubdetPM.initialize(_name,
                           "fC",
                           hcaldqm::hashfunctions::fSubdetPM,
                           new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::ffC_10000),
                           new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),
                           0);
  _cSumQ_SubdetPM.initialize(_name,
                             "SumQ",
                             hcaldqm::hashfunctions::fSubdetPM,
                             new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::ffC_10000),
                             new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),
                             0);
  _cSumQ_depth.initialize(_name,
                          "SumQ",
                          hcaldqm::hashfunctions::fdepth,
                          new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fieta),
                          new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fiphi),
                          new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::ffC_10000),
                          0);
  _cSumQvsLS_SubdetPM.initialize(_name,
                                 "SumQvsLS",
                                 hcaldqm::hashfunctions::fSubdetPM,
                                 new hcaldqm::quantity::LumiSection(_maxLS),
                                 new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::ffC_10000),
                                 0);
  _cTimingCut_SubdetPM.initialize(_name,
                                  "TimingCut",
                                  hcaldqm::hashfunctions::fSubdetPM,
                                  new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS200),
                                  new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                                  0);
  _cTimingCut_depth.initialize(_name,
                               "TimingCut",
                               hcaldqm::hashfunctions::fdepth,
                               new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fieta),
                               new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fiphi),
                               new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS200),
                               0);

  //  Occupancy w/o a cut
  _cOccupancyvsLS_Subdet.initialize(_name,
                                    "OccupancyvsLS",
                                    hcaldqm::hashfunctions::fSubdet,
                                    new hcaldqm::quantity::LumiSection(_maxLS),
                                    new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN_to8000),
                                    0);
  _cOccupancy_depth.initialize(_name,
                               "Occupancy",
                               hcaldqm::hashfunctions::fdepth,
                               new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fieta),
                               new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fiphi),
                               new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                               0);

  //  Occupancy w/ a cut
  _cOccupancyCutvsLS_Subdet.initialize(_name,
                                       "OccupancyCutvsLS",
                                       hcaldqm::hashfunctions::fSubdet,
                                       new hcaldqm::quantity::LumiSection(_maxLS),
                                       new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN_to8000),
                                       0);
  _cOccupancyCut_depth.initialize(_name,
                                  "OccupancyCut",
                                  hcaldqm::hashfunctions::fdepth,
                                  new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fieta),
                                  new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fiphi),
                                  new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                                  0);

  if (_ptype != fOffline) {  // hidefed2crate
    std::vector<int> vFEDs = hcaldqm::utilities::getFEDList(_emap);
    std::vector<int> vFEDsVME = hcaldqm::utilities::getFEDVMEList(_emap);
    std::vector<int> vFEDsuTCA = hcaldqm::utilities::getFEDuTCAList(_emap);

    std::vector<uint32_t> vFEDHF;
    vFEDHF.push_back(HcalElectronicsId(22, SLOT_uTCA_MIN, FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
    vFEDHF.push_back(HcalElectronicsId(29, SLOT_uTCA_MIN, FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
    vFEDHF.push_back(HcalElectronicsId(32, SLOT_uTCA_MIN, FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
    //  initialize filters
    _filter_FEDHF.initialize(filter::fPreserver, hcaldqm::hashfunctions::fFED, vFEDHF);

    //  push the rawIds of each fed into the vector...
    for (std::vector<int>::const_iterator it = vFEDsVME.begin(); it != vFEDsVME.end(); ++it) {
      _vhashFEDs.push_back(
          HcalElectronicsId(constants::FIBERCH_MIN, FIBER_VME_MIN, SPIGOT_MIN, (*it) - FED_VME_MIN).rawId());
    }
    for (std::vector<int>::const_iterator it = vFEDsuTCA.begin(); it != vFEDsuTCA.end(); ++it) {
      std::pair<uint16_t, uint16_t> cspair = hcaldqm::utilities::fed2crate(*it);
      _vhashFEDs.push_back(HcalElectronicsId(cspair.first, cspair.second, FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
    }

    _cShapeCut_FED.initialize(_name,
                              "Shape",
                              hcaldqm::hashfunctions::fFED,
                              new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS),
                              new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::ffC_10000),
                              0);

    _cTimingCut_FEDVME.initialize(_name,
                                  "TimingCut",
                                  hcaldqm::hashfunctions::fFED,
                                  new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSpigot),
                                  new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fFiberVMEFiberCh),
                                  new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS200),
                                  0);
    _cTimingCut_FEDuTCA.initialize(_name,
                                   "TimingCut",
                                   hcaldqm::hashfunctions::fFED,
                                   new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSlotuTCA),
                                   new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fFiberuTCAFiberCh),
                                   new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS200),
                                   0);
    _cTimingCut_ElectronicsVME.initialize(_name,
                                          "TimingCut",
                                          hcaldqm::hashfunctions::fElectronics,
                                          new hcaldqm::quantity::FEDQuantity(vFEDsVME),
                                          new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSpigot),
                                          new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS200),
                                          0);
    _cTimingCut_ElectronicsuTCA.initialize(_name,
                                           "TimingCut",
                                           hcaldqm::hashfunctions::fElectronics,
                                           new hcaldqm::quantity::FEDQuantity(vFEDsuTCA),
                                           new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSlotuTCA),
                                           new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS200),
                                           0);
    _cTimingCutvsLS_FED.initialize(_name,
                                   "TimingvsLS",
                                   hcaldqm::hashfunctions::fFED,
                                   new hcaldqm::quantity::LumiSection(_maxLS),
                                   new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS200),
                                   0);

    _cOccupancy_FEDVME.initialize(_name,
                                  "Occupancy",
                                  hcaldqm::hashfunctions::fFED,
                                  new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSpigot),
                                  new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fFiberVMEFiberCh),
                                  new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                                  0);
    _cOccupancy_FEDuTCA.initialize(_name,
                                   "Occupancy",
                                   hcaldqm::hashfunctions::fFED,
                                   new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSlotuTCA),
                                   new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fFiberuTCAFiberCh),
                                   new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                                   0);
    _cOccupancy_ElectronicsVME.initialize(_name,
                                          "Occupancy",
                                          hcaldqm::hashfunctions::fElectronics,
                                          new hcaldqm::quantity::FEDQuantity(vFEDsVME),
                                          new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSpigot),
                                          new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                                          0);
    _cOccupancy_ElectronicsuTCA.initialize(_name,
                                           "Occupancy",
                                           hcaldqm::hashfunctions::fElectronics,
                                           new hcaldqm::quantity::FEDQuantity(vFEDsuTCA),
                                           new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSlotuTCA),
                                           new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                                           0);

    _cOccupancyCut_FEDVME.initialize(_name,
                                     "OccupancyCut",
                                     hcaldqm::hashfunctions::fFED,
                                     new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSpigot),
                                     new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fFiberVMEFiberCh),
                                     new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                                     0);
    _cOccupancyCut_FEDuTCA.initialize(_name,
                                      "OccupancyCut",
                                      hcaldqm::hashfunctions::fFED,
                                      new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSlotuTCA),
                                      new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fFiberuTCAFiberCh),
                                      new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                                      0);
    _cOccupancyCut_ElectronicsVME.initialize(_name,
                                             "OccupancyCut",
                                             hcaldqm::hashfunctions::fElectronics,
                                             new hcaldqm::quantity::FEDQuantity(vFEDsVME),
                                             new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSpigot),
                                             new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                                             0);
    _cOccupancyCut_ElectronicsuTCA.initialize(_name,
                                              "OccupancyCut",
                                              hcaldqm::hashfunctions::fElectronics,
                                              new hcaldqm::quantity::FEDQuantity(vFEDsuTCA),
                                              new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSlotuTCA),
                                              new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                                              0);

    _cDigiSize_FED.initialize(_name,
                              "DigiSize",
                              hcaldqm::hashfunctions::fFED,
                              new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fDigiSize),
                              new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                              0);
  }

  //  BOOK HISTOGRAMS
  char cutstr[200];
  sprintf(cutstr, "_SumQHBHE%dHO%dHF%d", int(_cutSumQ_HBHE), int(_cutSumQ_HO), int(_cutSumQ_HF));
  char cutstr2[200];
  sprintf(cutstr2, "_SumQHF%d", int(_cutSumQ_HF));

  _cADC_SubdetPM.book(ib, _emap, _subsystem);

  _cfC_SubdetPM.book(ib, _emap, _subsystem);
  _cSumQ_SubdetPM.book(ib, _emap, _subsystem);
  _cSumQ_depth.book(ib, _emap, _subsystem);
  _cSumQvsLS_SubdetPM.book(ib, _emap, _subsystem);

  _cTimingCut_SubdetPM.book(ib, _emap, _subsystem);
  _cTimingCut_depth.book(ib, _emap, _subsystem);

  _cOccupancyvsLS_Subdet.book(ib, _emap, _subsystem);
  _cOccupancyCut_depth.book(ib, _emap, _subsystem);
  if (_ptype != fOffline)
    _cOccupancy_depth.book(ib, _emap, _subsystem);

  if (_ptype != fOffline) {  // hidefed2crate
    _cShapeCut_FED.book(ib, _emap, _subsystem);
    _cTimingCut_FEDVME.book(ib, _emap, _filter_uTCA, _subsystem);
    _cTimingCut_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
    _cTimingCutvsLS_FED.book(ib, _emap, _subsystem);
    _cTimingCut_ElectronicsVME.book(ib, _emap, _filter_uTCA, _subsystem);
    _cTimingCut_ElectronicsuTCA.book(ib, _emap, _filter_VME, _subsystem);
    _cOccupancy_FEDVME.book(ib, _emap, _filter_uTCA, _subsystem);
    _cOccupancy_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
    _cOccupancy_ElectronicsVME.book(ib, _emap, _filter_uTCA, _subsystem);
    _cOccupancy_ElectronicsuTCA.book(ib, _emap, _filter_VME, _subsystem);
    _cOccupancyCut_FEDVME.book(ib, _emap, _filter_uTCA, _subsystem);
    _cOccupancyCut_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
    _cOccupancyCut_ElectronicsVME.book(ib, _emap, _filter_uTCA, _subsystem);
    _cOccupancyCut_ElectronicsuTCA.book(ib, _emap, _filter_VME, _subsystem);
    _cDigiSize_FED.book(ib, _emap, _subsystem);
  }

  //  BOOK HISTOGRAMS that are only for Online
  _ehashmap.initialize(_emap, electronicsmap::fD2EHashMap);
  _dhashmap.initialize(_emap, electronicsmap::fE2DHashMap);

  //      MARK THESE HISTOGRAMS AS LUMI BASED FOR OFFLINE PROCESSING
  auto scope = DQMStore::IBooker::UseLumiScope(ib);
  if (_ptype == fOffline) {
    //_cDigiSize_FED.setLumiFlag(); // hidefed2crate : FED stuff not available offline anymore, so this histogram doesn't make sense?
    _cOccupancy_depth.book(ib, _emap, _subsystem);
  }

  //      book Number of Events vs LS histogram
  ib.setCurrentFolder(_subsystem + "/RunInfo");
  meNumEvents1LS = ib.book1D("NumberOfEvents", "NumberOfEvents", 1, 0, 1);

  //      book the flag for unknown ids and the online guy as well
  ib.setCurrentFolder(_subsystem + "/" + _name);
  meUnknownIds1LS = ib.book1D("UnknownIds", "UnknownIds", 1, 0, 1);
  _unknownIdsPresent = false;
}

/* virtual */ void DigiPhase1Task::_resetMonitors(hcaldqm::UpdateFreq uf) {
  DQTask::_resetMonitors(uf);

  switch (uf) {
    case hcaldqm::f1LS:
      _unknownIdsPresent = false;
      break;
    case hcaldqm::f50LS:
      break;
    default:
      break;
  }
}

/* virtual */ void DigiPhase1Task::_process(edm::Event const& e, edm::EventSetup const&) {
  edm::Handle<QIE11DigiCollection> chbhe;
  edm::Handle<HODigiCollection> cho;
  edm::Handle<QIE10DigiCollection> chf;

  if (!e.getByToken(_tokHBHE, chbhe))
    _logger.dqmthrow("Collection HBHEDigiCollection isn't available" + _tagHBHE.label() + " " + _tagHBHE.instance());
  if (!e.getByToken(_tokHO, cho))
    _logger.dqmthrow("Collection HODigiCollection isn't available" + _tagHO.label() + " " + _tagHO.instance());
  if (!e.getByToken(_tokHF, chf))
    _logger.dqmthrow("Collection HF QIE10Collection isn't available" + _tagHF.label() + " " + _tagHF.instance());

  //  extract some info per event
  meNumEvents1LS->Fill(0.5);  // just increment

  //  To fill histograms outside of the loop, you need to determine if there were
  //  any valid det ids first
  uint32_t rawidValid = 0;
  uint32_t rawidHBValid = 0;
  uint32_t rawidHEValid = 0;

  //  HB collection
  int numChs = 0;
  int numChsCut = 0;
  int numChsHE = 0;
  int numChsCutHE = 0;
  for (QIE11DigiCollection::const_iterator it = chbhe->begin(); it != chbhe->end(); ++it) {
    QIE11DataFrame const& frame = *it;
    double sumQ = hcaldqm::utilities::sumQ_v10<QIE11DataFrame>(frame, 2.5, 0, frame.samples() - 1);

    //  Explicit check on the DetIds present in the Collection
    HcalDetId const& did = frame.detid();
    uint32_t rawid = _ehashmap.lookup(did);
    if (rawid == 0) {
      meUnknownIds1LS->Fill(1);
      _unknownIdsPresent = true;
      continue;
    }
    HcalElectronicsId const& eid(rawid);
    if (did.subdet() == HcalBarrel) {
      rawidHBValid = did.rawId();
    } else if (did.subdet() == HcalEndcap) {
      rawidHEValid = did.rawId();
    } else {
      // Skip non-HB or HE detids
      continue;
    }

    //  filter out channels that are masked out
    if (_xQuality.exists(did)) {
      HcalChannelStatus cs(did.rawId(), _xQuality.get(did));
      if (cs.isBitSet(HcalChannelStatus::HcalCellMask) || cs.isBitSet(HcalChannelStatus::HcalCellDead))
        continue;
    }

    _cSumQ_SubdetPM.fill(did, sumQ);
    _cOccupancy_depth.fill(did);

    if (_ptype != fOffline) {  // hidefed2crate
      _cDigiSize_FED.fill(eid, frame.samples());
    }
    if (eid.isVMEid()) {
      if (_ptype != fOffline) {  // hidefed2crate
        _cOccupancy_FEDVME.fill(eid);
        _cOccupancy_ElectronicsVME.fill(eid);
      }
    } else {
      if (_ptype != fOffline) {  // hidefed2crate
        _cOccupancy_FEDuTCA.fill(eid);
        _cOccupancy_ElectronicsuTCA.fill(eid);
      }
      /*
			if (!it->validate(0, it->size()))
			{
				_cCapIdRots_depth.fill(did);
				_cCapIdRots_FEDuTCA.fill(eid, 1);
			}*/
    }

    for (int i = 0; i < frame.samples(); i++) {
      _cADC_SubdetPM.fill(did, frame[i].adc());
      _cfC_SubdetPM.fill(did, constants::adc2fC[frame[i].adc()]);
      if (_ptype != fOffline) {  // hidefed2crate
        if (sumQ > _cutSumQ_HBHE)
          _cShapeCut_FED.fill(eid, i, constants::adc2fC[frame[i].adc()]);
      }
    }

    if (sumQ > _cutSumQ_HBHE) {
      double timing = hcaldqm::utilities::aveTS_v10<QIE11DataFrame>(frame, 2.5, 0, frame.samples() - 1);
      _cTimingCut_SubdetPM.fill(did, timing);
      _cTimingCut_depth.fill(did, timing);
      _cOccupancyCut_depth.fill(did);
      if (_ptype != fOffline) {  // hidefed2crate
        _cTimingCutvsLS_FED.fill(eid, _currentLS, timing);
      }
      _cSumQ_depth.fill(did, sumQ);
      _cSumQvsLS_SubdetPM.fill(did, _currentLS, sumQ);

      if (eid.isVMEid()) {
        if (_ptype != fOffline) {  // hidefed2crate
          _cTimingCut_FEDVME.fill(eid, timing);
          _cOccupancyCut_FEDVME.fill(eid);
          _cTimingCut_ElectronicsVME.fill(eid, timing);
          _cOccupancyCut_ElectronicsVME.fill(eid);
        }
      } else {
        if (_ptype != fOffline) {  // hidefed2crate
          _cTimingCut_FEDuTCA.fill(eid, timing);
          _cOccupancyCut_FEDuTCA.fill(eid);
          _cTimingCut_ElectronicsuTCA.fill(eid, timing);
          _cOccupancyCut_ElectronicsuTCA.fill(eid);
        }
      }
      did.subdet() == HcalBarrel ? numChsCut++ : numChsCutHE++;
    }
    did.subdet() == HcalBarrel ? numChs++ : numChsHE++;
  }

  if (rawidHBValid != 0 && rawidHEValid != 0) {
    _cOccupancyvsLS_Subdet.fill(HcalDetId(rawidHBValid), _currentLS, numChs);
    _cOccupancyvsLS_Subdet.fill(HcalDetId(rawidHEValid), _currentLS, numChsHE);
  }
  numChs = 0;
  numChsCut = 0;

  //  reset
  rawidValid = 0;

  //  HO collection
  for (HODigiCollection::const_iterator it = cho->begin(); it != cho->end(); ++it) {
    double sumQ = hcaldqm::utilities::sumQ<HODataFrame>(*it, 8.5, 0, it->size() - 1);

    //  Explicit check on the DetIds present in the Collection
    HcalDetId const& did = it->id();
    uint32_t rawid = _ehashmap.lookup(did);
    if (rawid == 0) {
      meUnknownIds1LS->Fill(1);
      _unknownIdsPresent = true;
      continue;
    }
    HcalElectronicsId const& eid(rawid);
    if (did.subdet() == HcalOuter)
      rawidValid = did.rawId();

    //  filter out channels that are masked out
    if (_xQuality.exists(did)) {
      HcalChannelStatus cs(did.rawId(), _xQuality.get(did));
      if (cs.isBitSet(HcalChannelStatus::HcalCellMask) || cs.isBitSet(HcalChannelStatus::HcalCellDead))
        continue;
    }

    _cSumQ_SubdetPM.fill(did, sumQ);
    _cOccupancy_depth.fill(did);
    if (_ptype != fOffline) {  // hidefed2crate
      _cDigiSize_FED.fill(eid, it->size());
    }
    if (eid.isVMEid()) {
      if (_ptype != fOffline) {  // hidefed2crate
        _cOccupancy_FEDVME.fill(eid);
        _cOccupancy_ElectronicsVME.fill(eid);
      }
      /*
			if (!it->validate(0, it->size()))
				_cCapIdRots_FEDVME.fill(eid, 1);
				*/
    } else {
      if (_ptype != fOffline) {  // hidefed2crate
        _cOccupancy_FEDuTCA.fill(eid);
        _cOccupancy_ElectronicsuTCA.fill(eid);
      }
      /*
			if (!it->validate(0, it->size()))
				_cCapIdRots_FEDuTCA.fill(eid, 1);*/
    }

    for (int i = 0; i < it->size(); i++) {
      _cADC_SubdetPM.fill(did, it->sample(i).adc());
      _cfC_SubdetPM.fill(did, it->sample(i).nominal_fC());
      if (_ptype != fOffline) {  // hidefed2crate
        if (sumQ > _cutSumQ_HO)
          _cShapeCut_FED.fill(eid, i, it->sample(i).nominal_fC());
      }
    }

    if (sumQ > _cutSumQ_HO) {
      double timing = hcaldqm::utilities::aveTS<HODataFrame>(*it, 8.5, 0, it->size() - 1);
      _cSumQ_depth.fill(did, sumQ);
      _cSumQvsLS_SubdetPM.fill(did, _currentLS, sumQ);
      _cOccupancyCut_depth.fill(did);
      _cTimingCut_SubdetPM.fill(did, timing);
      _cTimingCut_depth.fill(did, timing);
      if (_ptype != fOffline) {  // hidefed2crate
        _cTimingCutvsLS_FED.fill(eid, _currentLS, timing);
      }

      if (eid.isVMEid()) {
        if (_ptype != fOffline) {  // hidefed2crate
          _cTimingCut_FEDVME.fill(eid, timing);
          _cOccupancyCut_FEDVME.fill(eid);
          _cTimingCut_ElectronicsVME.fill(eid, timing);
          _cOccupancyCut_ElectronicsVME.fill(eid);
        }
      } else {
        if (_ptype != fOffline) {  // hidefed2crate
          _cTimingCut_FEDuTCA.fill(eid, timing);
          _cOccupancyCut_FEDuTCA.fill(eid);
          _cTimingCut_ElectronicsuTCA.fill(eid, timing);
          _cOccupancyCut_ElectronicsuTCA.fill(eid);
        }
      }
      numChsCut++;
    }
    numChs++;
  }

  if (rawidValid != 0) {
    _cOccupancyvsLS_Subdet.fill(HcalDetId(rawidValid), _currentLS, numChs);
  }
  numChs = 0;
  numChsCut = 0;

  //  reset
  rawidValid = 0;

  //  HF collection
  for (QIE10DigiCollection::const_iterator it = chf->begin(); it != chf->end(); ++it) {
    QIE10DataFrame frame = *it;
    double sumQ = hcaldqm::utilities::sumQ_v10<QIE10DataFrame>(frame, 2.5, 0, frame.samples() - 1);

    //  Explicit check on the DetIds present in the Collection
    HcalDetId const& did = it->id();
    uint32_t rawid = _ehashmap.lookup(did);
    if (rawid == 0) {
      meUnknownIds1LS->Fill(1);
      _unknownIdsPresent = true;
      continue;
    }
    HcalElectronicsId const& eid(rawid);
    if (did.subdet() == HcalForward) {
      rawidValid = did.rawId();
    } else {
      // Skip non-HF detids
      continue;
    }

    //  filter out channels that are masked out
    if (_xQuality.exists(did)) {
      HcalChannelStatus cs(did.rawId(), _xQuality.get(did));
      if (cs.isBitSet(HcalChannelStatus::HcalCellMask) || cs.isBitSet(HcalChannelStatus::HcalCellDead))
        continue;
    }

    _cSumQ_SubdetPM.fill(did, sumQ);
    _cOccupancy_depth.fill(did);
    if (_ptype != fOffline) {  // hidefed2crate
      _cDigiSize_FED.fill(eid, frame.samples());
    }
    if (eid.isVMEid()) {
      if (_ptype != fOffline) {  // hidefed2crate
        _cOccupancy_FEDVME.fill(eid);
        _cOccupancy_ElectronicsVME.fill(eid);
      }
      /*
			if (!it->validate(0, it->size()))
				_cCapIdRots_FEDVME.fill(eid, 1);*/
    } else {
      if (_ptype != fOffline) {  // hidefed2crate
        _cOccupancy_FEDuTCA.fill(eid);
        _cOccupancy_ElectronicsuTCA.fill(eid);
      }
      /*
			if (!it->validate(0, it->size()))
				_cCapIdRots_FEDuTCA.fill(eid, 1);*/
    }

    for (int i = 0; i < frame.samples(); i++) {
      _cADC_SubdetPM.fill(did, frame[i].adc());
      _cfC_SubdetPM.fill(did, constants::adc2fC[frame[i].adc()]);
      if (_ptype != fOffline) {  // hidefed2crate
        if (sumQ > _cutSumQ_HF)
          _cShapeCut_FED.fill(eid, i, constants::adc2fC[frame[i].adc()]);
      }
    }

    if (sumQ > _cutSumQ_HF) {
      double timing = hcaldqm::utilities::aveTS_v10<QIE10DataFrame>(frame, 2.5, 0, frame.samples() - 1);

      _cSumQ_depth.fill(did, sumQ);
      _cSumQvsLS_SubdetPM.fill(did, _currentLS, sumQ);

      _cTimingCut_SubdetPM.fill(did, timing);
      _cTimingCut_depth.fill(did, timing);
      if (_ptype != fOffline) {  // hidefed2crate
        _cTimingCutvsLS_FED.fill(eid, _currentLS, timing);
      }
      _cOccupancyCut_depth.fill(did);
      if (eid.isVMEid()) {
        if (_ptype != fOffline) {  // hidefed2crate
          _cTimingCut_FEDVME.fill(eid, timing);
          _cOccupancyCut_FEDVME.fill(eid);
          _cTimingCut_ElectronicsVME.fill(eid, timing);
          _cOccupancyCut_ElectronicsVME.fill(eid);
        }
      } else {
        if (_ptype != fOffline) {  // hidefed2crate
          _cTimingCut_FEDuTCA.fill(eid, timing);
          _cOccupancyCut_FEDuTCA.fill(eid);
          _cTimingCut_ElectronicsuTCA.fill(eid, timing);
          _cOccupancyCut_ElectronicsuTCA.fill(eid);
        }
      }
      numChsCut++;
    }
    numChs++;
  }

  if (rawidValid != 0) {
    _cOccupancyvsLS_Subdet.fill(HcalDetId(rawidValid), _currentLS, numChs);
  }
}

/* virtual */ void DigiPhase1Task::dqmBeginLuminosityBlock(edm::LuminosityBlock const& lb, edm::EventSetup const& es) {
  DQTask::dqmBeginLuminosityBlock(lb, es);
}

/* virtual */ void DigiPhase1Task::dqmEndLuminosityBlock(edm::LuminosityBlock const& lb, edm::EventSetup const& es) {
  DQTask::dqmEndLuminosityBlock(lb, es);
}

DEFINE_FWK_MODULE(DigiPhase1Task);
