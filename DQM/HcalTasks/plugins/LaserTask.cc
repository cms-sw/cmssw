#include "DQM/HcalTasks/interface/LaserTask.h"

using namespace hcaldqm;
using namespace hcaldqm::constants;
using namespace hcaldqm::filter;
LaserTask::LaserTask(edm::ParameterSet const& ps)
    : DQTask(ps), hcalDbServiceToken_(esConsumes<HcalDbService, HcalDbRecord, edm::Transition::BeginRun>()) {
  _nevents = ps.getUntrackedParameter<int>("nevents", 2000);

  //	tags
  _tagQIE11 = ps.getUntrackedParameter<edm::InputTag>("tagHE", edm::InputTag("hcalDigis"));
  _tagHO = ps.getUntrackedParameter<edm::InputTag>("tagHO", edm::InputTag("hcalDigis"));
  _tagQIE10 = ps.getUntrackedParameter<edm::InputTag>("tagHF", edm::InputTag("hcalDigis"));
  _taguMN = ps.getUntrackedParameter<edm::InputTag>("taguMN", edm::InputTag("hcalDigis"));
  _tagLaserMon = ps.getUntrackedParameter<edm::InputTag>("tagLaserMon", edm::InputTag("LASERMON"));
  _tokQIE11 = consumes<QIE11DigiCollection>(_tagQIE11);
  _tokHO = consumes<HODigiCollection>(_tagHO);
  _tokQIE10 = consumes<QIE10DigiCollection>(_tagQIE10);
  _tokuMN = consumes<HcalUMNioDigi>(_taguMN);
  _tokLaserMon = consumes<QIE10DigiCollection>(_tagLaserMon);

  _vflags.resize(nLaserFlag);
  _vflags[fBadTiming] = hcaldqm::flag::Flag("BadTiming");
  _vflags[fMissingLaserMon] = hcaldqm::flag::Flag("MissingLaserMon");

  //	constants
  _lowHBHE = ps.getUntrackedParameter<double>("lowHBHE", 50);
  _lowHO = ps.getUntrackedParameter<double>("lowHO", 100);
  _lowHF = ps.getUntrackedParameter<double>("lowHF", 50);
  _laserType = (uint32_t)ps.getUntrackedParameter<uint32_t>("laserType");

  // Laser mon digi ordering list
  _vLaserMonIPhi = ps.getUntrackedParameter<std::vector<int>>("vLaserMonIPhi");
  _laserMonIEta = ps.getUntrackedParameter<int>("laserMonIEta");
  _laserMonCBox = ps.getUntrackedParameter<int>("laserMonCBox");
  _laserMonDigiOverlap = ps.getUntrackedParameter<int>("laserMonDigiOverlap");
  _laserMonTS0 = ps.getUntrackedParameter<int>("laserMonTS0");
  _laserMonThreshold = ps.getUntrackedParameter<double>("laserMonThreshold", 1.e5);
  _thresh_frac_timingreflm = ps.getUntrackedParameter<double>("_thresh_frac_timingreflm", 0.01);
  _thresh_min_lmsumq = ps.getUntrackedParameter<double>("thresh_min_lmsumq", 50000.);

  std::vector<double> vTimingRangeHB =
      ps.getUntrackedParameter<std::vector<double>>("thresh_timingreflm_HB", std::vector<double>({-70, -10.}));
  std::vector<double> vTimingRangeHE =
      ps.getUntrackedParameter<std::vector<double>>("thresh_timingreflm_HE", std::vector<double>({-60., 0.}));
  std::vector<double> vTimingRangeHO =
      ps.getUntrackedParameter<std::vector<double>>("thresh_timingreflm_HO", std::vector<double>({-50., 20.}));
  std::vector<double> vTimingRangeHF =
      ps.getUntrackedParameter<std::vector<double>>("thresh_timingreflm_HF", std::vector<double>({-50., 20.}));
  _thresh_timingreflm[HcalBarrel] = std::make_pair(vTimingRangeHB[0], vTimingRangeHB[1]);
  _thresh_timingreflm[HcalEndcap] = std::make_pair(vTimingRangeHE[0], vTimingRangeHE[1]);
  _thresh_timingreflm[HcalOuter] = std::make_pair(vTimingRangeHO[0], vTimingRangeHO[1]);
  _thresh_timingreflm[HcalForward] = std::make_pair(vTimingRangeHF[0], vTimingRangeHF[1]);
}

/* virtual */ void LaserTask::bookHistograms(DQMStore::IBooker& ib, edm::Run const& r, edm::EventSetup const& es) {
  if (_ptype == fLocal)
    if (r.runAuxiliary().run() == 1)
      return;

  DQTask::bookHistograms(ib, r, es);

  edm::ESHandle<HcalDbService> dbService = es.getHandle(hcalDbServiceToken_);
  _emap = dbService->getHcalMapping();

  std::vector<uint32_t> vhashVME;
  std::vector<uint32_t> vhashuTCA;
  std::vector<uint32_t> vhashC36;
  vhashVME.push_back(
      HcalElectronicsId(constants::FIBERCH_MIN, constants::FIBER_VME_MIN, SPIGOT_MIN, CRATE_VME_MIN).rawId());
  vhashuTCA.push_back(HcalElectronicsId(CRATE_uTCA_MIN, SLOT_uTCA_MIN, FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
  _filter_VME.initialize(filter::fFilter, hcaldqm::hashfunctions::fElectronics, vhashVME);
  _filter_uTCA.initialize(filter::fFilter, hcaldqm::hashfunctions::fElectronics, vhashuTCA);

  //	INITIALIZE
  _cSignalMean_Subdet.initialize(_name,
                                 "SignalMean",
                                 hcaldqm::hashfunctions::fSubdet,
                                 new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE10fC_100000Coarse),
                                 new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),
                                 0);
  _cSignalRMS_Subdet.initialize(_name,
                                "SignalRMS",
                                hcaldqm::hashfunctions::fSubdet,
                                new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::ffC_1000),
                                new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),
                                0);
  _cTimingMean_Subdet.initialize(_name,
                                 "TimingMean",
                                 hcaldqm::hashfunctions::fSubdet,
                                 new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS200),
                                 new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),
                                 0);
  _cTimingRMS_Subdet.initialize(_name,
                                "TimingRMS",
                                hcaldqm::hashfunctions::fSubdet,
                                new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS200),
                                new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),
                                0);

  _cADC_SubdetPM.initialize(_name,
                            "ADC",
                            hcaldqm::hashfunctions::fSubdetPM,
                            new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fADC_128),
                            new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),
                            0);

  if (_ptype == fLocal) {  // hidefed2crate
    _cSignalMean_FEDuTCA.initialize(_name,
                                    "SignalMean",
                                    hcaldqm::hashfunctions::fFED,
                                    new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSlotuTCA),
                                    new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fFiberuTCAFiberCh),
                                    new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE10fC_100000Coarse),
                                    0);
    _cSignalRMS_FEDuTCA.initialize(_name,
                                   "SignalRMS",
                                   hcaldqm::hashfunctions::fFED,
                                   new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSlotuTCA),
                                   new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fFiberuTCAFiberCh),
                                   new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::ffC_3000),
                                   0);
    _cTimingMean_FEDuTCA.initialize(_name,
                                    "TimingMean",
                                    hcaldqm::hashfunctions::fFED,
                                    new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSlotuTCA),
                                    new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fFiberuTCAFiberCh),
                                    new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS200),
                                    0);
    _cTimingRMS_FEDuTCA.initialize(_name,
                                   "TimingRMS",
                                   hcaldqm::hashfunctions::fFED,
                                   new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSlotuTCA),
                                   new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fFiberuTCAFiberCh),
                                   new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS200),
                                   0);
    _cMissing_FEDuTCA.initialize(_name,
                                 "Missing",
                                 hcaldqm::hashfunctions::fFED,
                                 new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSlotuTCA),
                                 new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fFiberuTCAFiberCh),
                                 new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                                 0);

    _cShapeCut_FEDSlot.initialize(_name,
                                  "Shape",
                                  hcaldqm::hashfunctions::fFEDSlot,
                                  new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS),
                                  new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::ffC_3000),
                                  0);
  }
  _cTimingvsEvent_SubdetPM.initialize(_name,
                                      "TimingvsEvent",
                                      hcaldqm::hashfunctions::fSubdetPM,
                                      new hcaldqm::quantity::EventNumber(_nevents),
                                      new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS200),
                                      0);
  _cSignalvsEvent_SubdetPM.initialize(_name,
                                      "SignalvsEvent",
                                      hcaldqm::hashfunctions::fSubdetPM,
                                      new hcaldqm::quantity::EventNumber(_nevents),
                                      new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::ffC_3000),
                                      0);
  _cTimingvsLS_SubdetPM.initialize(_name,
                                   "TimingvsLS",
                                   hcaldqm::hashfunctions::fSubdetPM,
                                   new hcaldqm::quantity::LumiSection(_maxLS),
                                   new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS200),
                                   0);
  _cSignalvsLS_SubdetPM.initialize(_name,
                                   "SignalvsLS",
                                   hcaldqm::hashfunctions::fSubdetPM,
                                   new hcaldqm::quantity::LumiSection(_maxLS),
                                   new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::ffC_3000),
                                   0);
  _cTimingvsBX_SubdetPM.initialize(_name,
                                   "TimingvsBX",
                                   hcaldqm::hashfunctions::fSubdetPM,
                                   new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fBX),
                                   new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS200),
                                   0);
  _cSignalvsBX_SubdetPM.initialize(_name,
                                   "SignalvsBX",
                                   hcaldqm::hashfunctions::fSubdetPM,
                                   new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fBX),
                                   new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::ffC_3000),
                                   0);

  _cSignalMean_depth.initialize(_name,
                                "SignalMean",
                                hcaldqm::hashfunctions::fdepth,
                                new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fieta),
                                new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fiphi),
                                new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE10fC_100000Coarse),
                                0);
  _cSignalRMS_depth.initialize(_name,
                               "SignalRMS",
                               hcaldqm::hashfunctions::fdepth,
                               new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fieta),
                               new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fiphi),
                               new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::ffC_1000),
                               0);
  _cTimingMean_depth.initialize(_name,
                                "TimingMean",
                                hcaldqm::hashfunctions::fdepth,
                                new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fieta),
                                new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fiphi),
                                new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS200),
                                0);
  _cTimingRMS_depth.initialize(_name,
                               "TimingRMS",
                               hcaldqm::hashfunctions::fdepth,
                               new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fieta),
                               new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fiphi),
                               new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS200),
                               0);

  _cMissing_depth.initialize(_name,
                             "Missing",
                             hcaldqm::hashfunctions::fdepth,
                             new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fieta),
                             new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fiphi),
                             new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                             0);

  //	initialize compact containers
  _xSignalSum.initialize(hcaldqm::hashfunctions::fDChannel);
  _xSignalSum2.initialize(hcaldqm::hashfunctions::fDChannel);
  _xTimingSum.initialize(hcaldqm::hashfunctions::fDChannel);
  _xTimingSum2.initialize(hcaldqm::hashfunctions::fDChannel);
  _xEntries.initialize(hcaldqm::hashfunctions::fDChannel);

  // LaserMon containers
  if (_ptype == fOnline || _ptype == fLocal) {
    _cLaserMonSumQ.initialize(_name,
                              "LaserMonSumQ",
                              new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::ffC_1000000),
                              new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                              0);
    _cLaserMonSumQ.showOverflowX(true);

    _cLaserMonTiming.initialize(_name,
                                "LaserMonTiming",
                                new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTime_ns_250),
                                new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                                0);
    _cLaserMonTiming.showOverflowX(true);

    if (_ptype == fOnline) {
      _cLaserMonSumQ_LS.initialize(_name,
                                   "LaserMonSumQ_LS",
                                   new hcaldqm::quantity::LumiSectionCoarse(_maxLS, 10),
                                   new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::ffC_1000000));
      _cLaserMonTiming_LS.initialize(_name,
                                     "LaserMonTiming_LS",
                                     new hcaldqm::quantity::LumiSectionCoarse(_maxLS, 10),
                                     new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_100TS));
      _cTimingDiffLS_SubdetPM.initialize(_name,
                                         "TimingDiff_DigiMinusLaserMon",
                                         hcaldqm::hashfunctions::fSubdetPM,
                                         new hcaldqm::quantity::LumiSectionCoarse(_maxLS, 10),
                                         new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fRBX),
                                         new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTimingDiff_ns),
                                         0);
      _cTimingDiffLS_SubdetPM.showOverflowY(true);
    } else if (_ptype == fLocal) {
      _cLaserMonSumQ_Event.initialize(_name,
                                      "LaserMonSumQ_Event",
                                      new hcaldqm::quantity::EventNumber(_nevents),
                                      new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::ffC_1000000));
      _cLaserMonTiming_Event.initialize(_name,
                                        "LaserMonTiming_Event",
                                        new hcaldqm::quantity::EventNumber(_nevents),
                                        new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_100TS));
      _cTimingDiffEvent_SubdetPM.initialize(_name,
                                            "TimingDiff_DigiMinusLaserMon",
                                            hcaldqm::hashfunctions::fSubdetPM,
                                            new hcaldqm::quantity::EventNumber(_nevents),
                                            new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fRBX),
                                            new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTimingDiff_ns),
                                            0);
      _cTimingDiffEvent_SubdetPM.showOverflowY(true);
    }
    _cTiming_DigivsLaserMon_SubdetPM.initialize(
        _name,
        "Timing_DigivsLaserMon",
        hcaldqm::hashfunctions::fSubdetPM,
        new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTime_ns_250_coarse),
        new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTime_ns_250_coarse),
        new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
        0);
    _cTiming_DigivsLaserMon_SubdetPM.showOverflowX(true);
    _cTiming_DigivsLaserMon_SubdetPM.showOverflowY(true);

    _xTimingRefLMSum.initialize(hcaldqm::hashfunctions::fDChannel);
    _xTimingRefLMSum2.initialize(hcaldqm::hashfunctions::fDChannel);
    _xNBadTimingRefLM.initialize(hcaldqm::hashfunctions::fFED);
    _xNChs.initialize(hcaldqm::hashfunctions::fFED);
    _xMissingLaserMon = 0;

    std::vector<int> vFEDs = hcaldqm::utilities::getFEDList(_emap);
    std::vector<int> vFEDsVME = hcaldqm::utilities::getFEDVMEList(_emap);
    std::vector<int> vFEDsuTCA = hcaldqm::utilities::getFEDuTCAList(_emap);
    for (std::vector<int>::const_iterator it = vFEDsVME.begin(); it != vFEDsVME.end(); ++it)
      _vhashFEDs.push_back(
          HcalElectronicsId(constants::FIBERCH_MIN, FIBER_VME_MIN, SPIGOT_MIN, (*it) - FED_VME_MIN).rawId());
    for (std::vector<int>::const_iterator it = vFEDsuTCA.begin(); it != vFEDsuTCA.end(); ++it) {
      std::pair<uint16_t, uint16_t> cspair = utilities::fed2crate(*it);
      _vhashFEDs.push_back(HcalElectronicsId(cspair.first, cspair.second, FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
    }

    _cSummaryvsLS_FED.initialize(_name,
                                 "SummaryvsLS",
                                 hcaldqm::hashfunctions::fFED,
                                 new hcaldqm::quantity::LumiSection(_maxLS),
                                 new hcaldqm::quantity::FlagQuantity(_vflags),
                                 new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fState),
                                 0);
    _cSummaryvsLS.initialize(_name,
                             "SummaryvsLS",
                             new hcaldqm::quantity::LumiSection(_maxLS),
                             new hcaldqm::quantity::FEDQuantity(vFEDs),
                             new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fState),
                             0);
  }  // End if (_ptype == fOnline || _ptype == fLocal) {

  //	BOOK
  _cSignalMean_Subdet.book(ib, _emap, _subsystem);
  _cSignalRMS_Subdet.book(ib, _emap, _subsystem);
  _cTimingMean_Subdet.book(ib, _emap, _subsystem);
  _cTimingRMS_Subdet.book(ib, _emap, _subsystem);

  _cSignalMean_depth.book(ib, _emap, _subsystem);
  _cSignalRMS_depth.book(ib, _emap, _subsystem);
  _cTimingMean_depth.book(ib, _emap, _subsystem);
  _cTimingRMS_depth.book(ib, _emap, _subsystem);

  if (_ptype == fLocal) {
    _cTimingvsEvent_SubdetPM.book(ib, _emap, _subsystem);
    _cSignalvsEvent_SubdetPM.book(ib, _emap, _subsystem);
  } else {
    _cTimingvsLS_SubdetPM.book(ib, _emap, _subsystem);
    _cSignalvsLS_SubdetPM.book(ib, _emap, _subsystem);
    _cTimingvsBX_SubdetPM.book(ib, _emap, _subsystem);
    _cSignalvsBX_SubdetPM.book(ib, _emap, _subsystem);
  }

  if (_ptype == fLocal) {  // hidefed2crate
    _cSignalMean_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
    _cSignalRMS_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
    _cTimingMean_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
    _cTimingRMS_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
    _cMissing_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
    _cShapeCut_FEDSlot.book(ib, _emap, _subsystem);
  }
  _cADC_SubdetPM.book(ib, _emap, _subsystem);

  _cMissing_depth.book(ib, _emap, _subsystem);

  _xSignalSum.book(_emap);
  _xSignalSum2.book(_emap);
  _xEntries.book(_emap);
  _xTimingSum.book(_emap);
  _xTimingSum2.book(_emap);

  if (_ptype == fOnline || _ptype == fLocal) {
    _cLaserMonSumQ.book(ib, _subsystem);
    _cLaserMonTiming.book(ib, _subsystem);
    if (_ptype == fOnline) {
      _cLaserMonSumQ_LS.book(ib, _subsystem);
      _cLaserMonTiming_LS.book(ib, _subsystem);
      _cTimingDiffLS_SubdetPM.book(ib, _emap, _subsystem);
    } else if (_ptype == fLocal) {
      _cLaserMonSumQ_Event.book(ib, _subsystem);
      _cLaserMonTiming_Event.book(ib, _subsystem);
      _cTimingDiffEvent_SubdetPM.book(ib, _emap, _subsystem);
    }
    _cTiming_DigivsLaserMon_SubdetPM.book(ib, _emap, _subsystem);
    _xTimingRefLMSum.book(_emap);
    _xTimingRefLMSum2.book(_emap);
    _xNBadTimingRefLM.book(_emap);
    _xNChs.book(_emap);
    _cSummaryvsLS_FED.book(ib, _emap, _subsystem);
    _cSummaryvsLS.book(ib, _subsystem);
  }

  _ehashmap.initialize(_emap, electronicsmap::fD2EHashMap);
}

/* virtual */ void LaserTask::_resetMonitors(hcaldqm::UpdateFreq uf) { DQTask::_resetMonitors(uf); }

/* virtual */ void LaserTask::_dump() {
  _cSignalMean_Subdet.reset();
  _cSignalRMS_Subdet.reset();
  _cTimingMean_Subdet.reset();
  _cTimingRMS_Subdet.reset();
  _cSignalMean_depth.reset();
  _cSignalRMS_depth.reset();
  _cTimingMean_depth.reset();
  _cTimingRMS_depth.reset();

  if (_ptype == fLocal) {  // hidefed2crate
    _cSignalMean_FEDuTCA.reset();
    _cSignalRMS_FEDuTCA.reset();
    _cTimingMean_FEDuTCA.reset();
    _cTimingRMS_FEDuTCA.reset();
  }
  if (_ptype != fOffline) {
    _xNChs.reset();
  }

  std::vector<HcalGenericDetId> dids = _emap->allPrecisionId();
  for (std::vector<HcalGenericDetId>::const_iterator it = dids.begin(); it != dids.end(); ++it) {
    if (!it->isHcalDetId())
      continue;
    HcalDetId did = HcalDetId(it->rawId());
    HcalElectronicsId eid(_ehashmap.lookup(*it));
    int n = _xEntries.get(did);
    //	channels missing or low signal
    if (n == 0) {
      _cMissing_depth.fill(did);
      if (_ptype == fLocal) {  // hidefed2crate
        if (!eid.isVMEid())
          _cMissing_FEDuTCA.fill(eid);
      }
      continue;
    }

    _xNChs.get(eid)++;

    double msig = _xSignalSum.get(did) / n;
    double mtim = _xTimingSum.get(did) / n;
    double rsig = sqrt(_xSignalSum2.get(did) / n - msig * msig);
    double rtim = sqrt(_xTimingSum2.get(did) / n - mtim * mtim);

    _cSignalMean_Subdet.fill(did, msig);
    _cSignalMean_depth.fill(did, msig);
    _cSignalRMS_Subdet.fill(did, rsig);
    _cSignalRMS_depth.fill(did, rsig);
    _cTimingMean_Subdet.fill(did, mtim);
    _cTimingMean_depth.fill(did, mtim);
    _cTimingRMS_Subdet.fill(did, rtim);
    _cTimingRMS_depth.fill(did, rtim);
    if (_ptype == fLocal) {  // hidefed2crate
      if (!eid.isVMEid()) {
        _cSignalMean_FEDuTCA.fill(eid, msig);
        _cSignalRMS_FEDuTCA.fill(eid, rsig);
        _cTimingMean_FEDuTCA.fill(eid, mtim);
        _cTimingRMS_FEDuTCA.fill(eid, rtim);
      }
    }

    // Bad timing

    double timingreflm_mean = _xTimingRefLMSum.get(did) / n;
    //double timingreflm_rms = sqrt(_xTimingRefLMSum2.get(did) / n - timingreflm_mean * timingreflm_mean);

    if ((timingreflm_mean < _thresh_timingreflm[did.subdet()].first) ||
        (timingreflm_mean > _thresh_timingreflm[did.subdet()].second)) {
      _xNBadTimingRefLM.get(eid)++;
    }
  }
  if (_ptype != fOffline) {  // hidefed2crate
    for (std::vector<uint32_t>::const_iterator it = _vhashFEDs.begin(); it != _vhashFEDs.end(); ++it) {
      hcaldqm::flag::Flag fSum("LASER");
      HcalElectronicsId eid = HcalElectronicsId(*it);
      std::vector<uint32_t>::const_iterator jt = std::find(_vcdaqEids.begin(), _vcdaqEids.end(), (*it));
      if (jt == _vcdaqEids.end()) {
        //	not @cDAQ
        for (uint32_t iflag = 0; iflag < _vflags.size(); iflag++)
          _cSummaryvsLS_FED.setBinContent(eid, _currentLS, int(iflag), int(hcaldqm::flag::fNCDAQ));
        _cSummaryvsLS.setBinContent(eid, _currentLS, int(hcaldqm::flag::fNCDAQ));
        continue;
      }
      //	@cDAQ
      if (hcaldqm::utilities::isFEDHBHE(eid) || hcaldqm::utilities::isFEDHO(eid) || hcaldqm::utilities::isFEDHF(eid)) {
        int min_nchs = 10;
        if (_xNChs.get(eid) < min_nchs) {
          _vflags[fBadTiming]._state = hcaldqm::flag::fNA;
        } else {
          double frbadtimingreflm = double(_xNBadTimingRefLM.get(eid)) / double(_xNChs.get(eid));
          if (frbadtimingreflm > _thresh_frac_timingreflm) {
            _vflags[fBadTiming]._state = hcaldqm::flag::fBAD;
          } else {
            _vflags[fBadTiming]._state = hcaldqm::flag::fGOOD;
          }
        }
        if (_xMissingLaserMon) {
          _vflags[fMissingLaserMon]._state = hcaldqm::flag::fBAD;
        } else {
          _vflags[fMissingLaserMon]._state = hcaldqm::flag::fGOOD;
        }
      }

      // Set SummaryVsLS bins
      int iflag = 0;
      for (std::vector<hcaldqm::flag::Flag>::iterator ft = _vflags.begin(); ft != _vflags.end(); ++ft) {
        _cSummaryvsLS_FED.setBinContent(eid, _currentLS, iflag, int(ft->_state));
        fSum += (*ft);
        iflag++;
        ft->reset();
      }
      _cSummaryvsLS.setBinContent(eid, _currentLS, fSum._state);
    }  // End loop over FEDs

    // Reset per-LS containers
    _xNBadTimingRefLM.reset();
    _xMissingLaserMon = 0;
  }  // End if _ptype != fOffline
}

/* virtual */ void LaserTask::_process(edm::Event const& e, edm::EventSetup const& es) {
  edm::Handle<QIE11DigiCollection> c_QIE11;
  edm::Handle<HODigiCollection> c_ho;
  edm::Handle<QIE10DigiCollection> c_QIE10;

  if (!e.getByToken(_tokQIE11, c_QIE11))
    _logger.dqmthrow("Collection QIE11DigiCollection isn't available " + _tagQIE11.label() + " " +
                     _tagQIE11.instance());
  if (!e.getByToken(_tokHO, c_ho))
    _logger.dqmthrow("Collection HODigiCollection isn't available " + _tagHO.label() + " " + _tagHO.instance());
  if (!e.getByToken(_tokQIE10, c_QIE10))
    _logger.dqmthrow("Collection QIE10DigiCollection isn't available " + _tagQIE10.label() + " " +
                     _tagQIE10.instance());

  //	int currentEvent = e.eventAuxiliary().id().event();
  int bx = e.bunchCrossing();

  // LASERMON
  edm::Handle<QIE10DigiCollection> cLaserMon;
  if (!e.getByToken(_tokLaserMon, cLaserMon)) {
    _logger.dqmthrow("QIE10DigiCollection for laserMonDigis isn't available.");
  }
  std::vector<int> laserMonADC;
  processLaserMon(cLaserMon, laserMonADC);

  // SumQ = peak +/- 3 TSes
  // Timing = fC-weighted average (TS-TS0) * 25 ns, also in peak +/- 3 TSes
  int peakTS = -1;
  double peakLaserMonADC = -1;
  for (unsigned int iTS = 0; iTS < laserMonADC.size(); ++iTS) {
    if (laserMonADC[iTS] > peakLaserMonADC) {
      peakLaserMonADC = laserMonADC[iTS];
      peakTS = iTS;
    }
  }

  double laserMonSumQ = 0;
  double laserMonTiming = 0.;

  if (peakTS >= 0) {
    int minTS = std::max(0, peakTS - 3);
    int maxTS = std::min(int(laserMonADC.size() - 1), peakTS + 3);
    for (int iTS = minTS; iTS <= maxTS; ++iTS) {
      double this_fC = hcaldqm::constants::adc2fC[laserMonADC[iTS]];
      laserMonSumQ += this_fC;
      laserMonTiming += 25. * (iTS - _laserMonTS0) * this_fC;
    }
  }
  if (laserMonSumQ > 0.) {
    laserMonTiming = laserMonTiming / laserMonSumQ;
  } else {
    ++_xMissingLaserMon;
  }

  if (laserMonSumQ > _laserMonThreshold) {
    _cLaserMonSumQ.fill(laserMonSumQ);
    _cLaserMonTiming.fill(laserMonTiming);
    if (_ptype == fOnline) {
      _cLaserMonSumQ_LS.fill(_currentLS, laserMonSumQ);
      _cLaserMonTiming_LS.fill(_currentLS, laserMonTiming);
    } else if (_ptype == fLocal) {
      int currentEvent = e.eventAuxiliary().id().event();
      _cLaserMonSumQ_Event.fill(currentEvent, laserMonSumQ);
      _cLaserMonTiming_Event.fill(currentEvent, laserMonTiming);
    }
  }

  for (QIE11DigiCollection::const_iterator it = c_QIE11->begin(); it != c_QIE11->end(); ++it) {
    const QIE11DataFrame digi = static_cast<const QIE11DataFrame>(*it);
    HcalDetId const& did = digi.detid();
    if ((did.subdet() != HcalBarrel) && (did.subdet() != HcalEndcap)) {
      continue;
    }
    uint32_t rawid = _ehashmap.lookup(did);
    HcalElectronicsId const& eid(rawid);

    CaloSamples digi_fC = hcaldqm::utilities::loadADC2fCDB<QIE11DataFrame>(_dbService, did, digi);
    //double sumQ = hcaldqm::utilities::sumQ_v10<QIE11DataFrame>(digi, 2.5, 0, digi.samples()-1);
    double sumQ = hcaldqm::utilities::sumQDB<QIE11DataFrame>(_dbService, digi_fC, did, digi, 0, digi.samples() - 1);
    if (sumQ < _lowHBHE)
      continue;

    //double aveTS = hcaldqm::utilities::aveTS_v10<QIE11DataFrame>(digi, 2.5, 0,digi.samples()-1);
    double aveTS = hcaldqm::utilities::aveTSDB<QIE11DataFrame>(_dbService, digi_fC, did, digi, 0, digi.samples() - 1);
    _xSignalSum.get(did) += sumQ;
    _xSignalSum2.get(did) += sumQ * sumQ;
    _xTimingSum.get(did) += aveTS;
    _xTimingSum2.get(did) += aveTS * aveTS;
    _xEntries.get(did)++;

    for (int i = 0; i < digi.samples(); i++) {
      if (_ptype == fLocal) {
        _cShapeCut_FEDSlot.fill(
            eid, i, hcaldqm::utilities::adc2fCDBMinusPedestal<QIE11DataFrame>(_dbService, digi_fC, did, digi, i));
      }
      _cADC_SubdetPM.fill(did, digi[i].adc());
    }

    //	select based on local global
    double digiTimingSOI = (aveTS - digi.presamples()) * 25.;
    double deltaTiming = digiTimingSOI - laserMonTiming;
    _cTiming_DigivsLaserMon_SubdetPM.fill(did, laserMonTiming, digiTimingSOI);
    _xTimingRefLMSum.get(did) += deltaTiming;
    _xTimingRefLMSum2.get(did) += deltaTiming * deltaTiming;
    if (_ptype == fLocal) {
      int currentEvent = e.eventAuxiliary().id().event();
      _cTimingvsEvent_SubdetPM.fill(did, currentEvent, aveTS);
      _cSignalvsEvent_SubdetPM.fill(did, currentEvent, sumQ);
      _cTimingDiffEvent_SubdetPM.fill(did, currentEvent, hcaldqm::utilities::getRBX(did.iphi()), deltaTiming);
    } else {
      _cTimingvsLS_SubdetPM.fill(did, _currentLS, aveTS);
      _cSignalvsLS_SubdetPM.fill(did, _currentLS, sumQ);
      _cTimingvsBX_SubdetPM.fill(did, bx, aveTS);
      _cSignalvsBX_SubdetPM.fill(did, bx, sumQ);
      _cTimingDiffLS_SubdetPM.fill(did, _currentLS, hcaldqm::utilities::getRBX(did.iphi()), deltaTiming);
    }
  }
  for (HODigiCollection::const_iterator it = c_ho->begin(); it != c_ho->end(); ++it) {
    const HODataFrame digi = (const HODataFrame)(*it);
    double sumQ = hcaldqm::utilities::sumQ<HODataFrame>(digi, 8.5, 0, digi.size() - 1);
    if (sumQ < _lowHO)
      continue;
    HcalDetId did = digi.id();
    HcalElectronicsId eid = digi.elecId();

    double aveTS = hcaldqm::utilities::aveTS<HODataFrame>(digi, 8.5, 0, digi.size() - 1);
    _xSignalSum.get(did) += sumQ;
    _xSignalSum2.get(did) += sumQ * sumQ;
    _xTimingSum.get(did) += aveTS;
    _xTimingSum2.get(did) += aveTS * aveTS;
    _xEntries.get(did)++;

    for (int i = 0; i < digi.size(); i++) {
      if (_ptype == fLocal) {  // hidefed2crate
        _cShapeCut_FEDSlot.fill(eid, i, digi.sample(i).nominal_fC() - 8.5);
      }
      _cADC_SubdetPM.fill(did, digi.sample(i).adc());
    }

    //	select based on local global
    double digiTimingSOI = (aveTS - digi.presamples()) * 25.;
    double deltaTiming = digiTimingSOI - laserMonTiming;
    _cTiming_DigivsLaserMon_SubdetPM.fill(did, laserMonTiming, digiTimingSOI);
    _xTimingRefLMSum.get(did) += deltaTiming;
    _xTimingRefLMSum2.get(did) += deltaTiming * deltaTiming;
    if (_ptype == fLocal) {
      int currentEvent = e.eventAuxiliary().id().event();
      _cTimingvsEvent_SubdetPM.fill(did, currentEvent, aveTS);
      _cSignalvsEvent_SubdetPM.fill(did, currentEvent, sumQ);
      _cTimingDiffEvent_SubdetPM.fill(did, currentEvent, hcaldqm::utilities::getRBX(did.iphi()), deltaTiming);
    } else {
      _cTimingvsLS_SubdetPM.fill(did, _currentLS, aveTS);
      _cSignalvsLS_SubdetPM.fill(did, _currentLS, sumQ);
      _cTimingvsBX_SubdetPM.fill(did, bx, aveTS);
      _cSignalvsBX_SubdetPM.fill(did, bx, sumQ);
      _cTimingDiffLS_SubdetPM.fill(did, _currentLS, hcaldqm::utilities::getRBX(did.iphi()), deltaTiming);
    }
  }
  for (QIE10DigiCollection::const_iterator it = c_QIE10->begin(); it != c_QIE10->end(); ++it) {
    const QIE10DataFrame digi = (const QIE10DataFrame)(*it);
    HcalDetId did = digi.detid();
    if (did.subdet() != HcalForward) {
      continue;
    }
    HcalElectronicsId eid = HcalElectronicsId(_ehashmap.lookup(did));

    CaloSamples digi_fC = hcaldqm::utilities::loadADC2fCDB<QIE10DataFrame>(_dbService, did, digi);
    double sumQ = hcaldqm::utilities::sumQDB<QIE10DataFrame>(_dbService, digi_fC, did, digi, 0, digi.samples() - 1);
    //double sumQ = hcaldqm::utilities::sumQ_v10<QIE10DataFrame>(digi, 2.5, 0, digi.samples()-1);
    if (sumQ < _lowHF)
      continue;

    //double aveTS = hcaldqm::utilities::aveTS_v10<QIE10DataFrame>(digi, 2.5, 0, digi.samples()-1);
    double aveTS = hcaldqm::utilities::aveTSDB<QIE10DataFrame>(_dbService, digi_fC, did, digi, 0, digi.samples() - 1);

    _xSignalSum.get(did) += sumQ;
    _xSignalSum2.get(did) += sumQ * sumQ;
    _xTimingSum.get(did) += aveTS;
    _xTimingSum2.get(did) += aveTS * aveTS;
    _xEntries.get(did)++;

    for (int i = 0; i < digi.samples(); i++) {
      if (_ptype == fLocal) {  // hidefed2crate
        _cShapeCut_FEDSlot.fill(
            eid, (int)i, hcaldqm::utilities::adc2fCDBMinusPedestal<QIE10DataFrame>(_dbService, digi_fC, did, digi, i));
      }
      _cADC_SubdetPM.fill(did, digi[i].adc());
    }

    //	select based on local global
    double digiTimingSOI = (aveTS - digi.presamples()) * 25.;
    double deltaTiming = digiTimingSOI - laserMonTiming;
    _cTiming_DigivsLaserMon_SubdetPM.fill(did, laserMonTiming, digiTimingSOI);
    _xTimingRefLMSum.get(did) += deltaTiming;
    _xTimingRefLMSum2.get(did) += deltaTiming * deltaTiming;
    if (_ptype == fLocal) {
      int currentEvent = e.eventAuxiliary().id().event();
      _cTimingvsEvent_SubdetPM.fill(did, currentEvent, aveTS);
      _cSignalvsEvent_SubdetPM.fill(did, currentEvent, sumQ);
      _cTimingDiffEvent_SubdetPM.fill(did, currentEvent, hcaldqm::utilities::getRBX(did.iphi()), deltaTiming);
    } else {
      _cTimingvsLS_SubdetPM.fill(did, _currentLS, aveTS);
      _cSignalvsLS_SubdetPM.fill(did, _currentLS, sumQ);
      _cTimingvsBX_SubdetPM.fill(did, bx, aveTS);
      _cSignalvsBX_SubdetPM.fill(did, bx, sumQ);
      _cTimingDiffLS_SubdetPM.fill(did, _currentLS, hcaldqm::utilities::getRBX(did.iphi()), deltaTiming);
    }
  }
}

void LaserTask::processLaserMon(edm::Handle<QIE10DigiCollection>& col, std::vector<int>& iLaserMonADC) {
  for (QIE10DigiCollection::const_iterator it = col->begin(); it != col->end(); ++it) {
    const QIE10DataFrame digi = (const QIE10DataFrame)(*it);
    HcalCalibDetId hcdid(digi.id());

    if ((hcdid.ieta() != _laserMonIEta) || (hcdid.cboxChannel() != _laserMonCBox)) {
      continue;
    }

    unsigned int digiIndex =
        std::find(_vLaserMonIPhi.begin(), _vLaserMonIPhi.end(), hcdid.iphi()) - _vLaserMonIPhi.begin();
    if (digiIndex == _vLaserMonIPhi.size()) {
      continue;
    }

    // First digi: initialize the vectors to -1 (need to know the length of the digi)
    if (iLaserMonADC.empty()) {
      int totalNSamples = (digi.samples() - _laserMonDigiOverlap) * _vLaserMonIPhi.size();
      for (int i = 0; i < totalNSamples; ++i) {
        iLaserMonADC.push_back(-1);
      }
    }

    for (int subindex = 0; subindex < digi.samples() - _laserMonDigiOverlap; ++subindex) {
      int totalIndex = (digi.samples() - _laserMonDigiOverlap) * digiIndex + subindex;
      iLaserMonADC[totalIndex] = (digi[subindex].ok() ? digi[subindex].adc() : -1);
    }
  }
}

/* virtual */ void LaserTask::globalEndLuminosityBlock(edm::LuminosityBlock const& lb, edm::EventSetup const& es) {
  auto lumiCache = luminosityBlockCache(lb.index());
  _currentLS = lumiCache->currentLS;

  if (_ptype == fLocal)
    return;
  this->_dump();

  DQTask::globalEndLuminosityBlock(lb, es);
}

/* virtual */ bool LaserTask::_isApplicable(edm::Event const& e) {
  if (_ptype != fOnline)
    return true;
  else {
    //	fOnline mode
    edm::Handle<HcalUMNioDigi> cumn;
    if (!e.getByToken(_tokuMN, cumn))
      return false;

    //	event type check first
    uint8_t eventType = cumn->eventType();
    if (eventType != constants::EVENTTYPE_LASER)
      return false;

    //	check if this analysis task is of the right laser type
    uint32_t laserType = cumn->valueUserWord(0);
    if (laserType == _laserType)
      return true;
  }

  return false;
}

DEFINE_FWK_MODULE(LaserTask);
