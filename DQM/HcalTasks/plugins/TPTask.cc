#include "DQM/HcalTasks/interface/TPTask.h"
#include "DQM/L1TMonitor/interface/L1TStage2CaloLayer1.h"  // For ComparisonHelper::zip

using namespace hcaldqm;
using namespace hcaldqm::constants;
TPTask::TPTask(edm::ParameterSet const& ps)
    : DQTask(ps), hcalDbServiceToken_(esConsumes<HcalDbService, HcalDbRecord, edm::Transition::BeginRun>()) {
  _tagData = ps.getUntrackedParameter<edm::InputTag>("tagData", edm::InputTag("hcalDigis"));
  _tagDataL1Rec = ps.getUntrackedParameter<edm::InputTag>("tagDataL1Rec", edm::InputTag("caloLayer1Digis"));
  _tagEmul = ps.getUntrackedParameter<edm::InputTag>("tagEmul", edm::InputTag("emulDigis"));
  _tagEmulNoTDCCut = ps.getUntrackedParameter<edm::InputTag>("tagEmulNoTDCCut", edm::InputTag("emulTPDigisNoTDCCut"));

  _tokData = consumes<HcalTrigPrimDigiCollection>(_tagData);
  _tokDataL1Rec = consumes<HcalTrigPrimDigiCollection>(_tagDataL1Rec);
  _tokEmul = consumes<HcalTrigPrimDigiCollection>(_tagEmul);
  _tokEmulNoTDCCut = consumes<HcalTrigPrimDigiCollection>(_tagEmulNoTDCCut);

  _skip1x1 = ps.getUntrackedParameter<bool>("skip1x1", true);
  _cutEt = ps.getUntrackedParameter<int>("cutEt", 3);
  _thresh_EtMsmRate_high = ps.getUntrackedParameter<double>("thresh_EtMsmRate_high", 0.2);
  _thresh_EtMsmRate_low = ps.getUntrackedParameter<double>("thresh_EtMsmRate_low", 0.05);
  _thresh_FGMsmRate_high = ps.getUntrackedParameter<double>("thresh_FGMsmRate", 0.2);
  _thresh_FGMsmRate_low = ps.getUntrackedParameter<double>("thresh_FGMsmRate_low", 0.05);
  _thresh_DataMsn = ps.getUntrackedParameter<double>("thresh_DataMsn", 0.1);
  _thresh_EmulMsn = ps.getUntrackedParameter<double>("thresh_EmulMsn");
  std::vector<int> tmp = ps.getUntrackedParameter<std::vector<int> >("vFGBitsReady");
  for (uint32_t iii = 0; iii < constants::NUM_FGBITS; iii++)
    _vFGBitsReady.push_back(tmp[iii]);

  _vflags.resize(nTPFlag);
  _vflags[fEtMsm] = flag::Flag("EtMsm");
  _vflags[fDataMsn] = flag::Flag("DataMsn");
  _vflags[fEmulMsn] = flag::Flag("EmulMsn");
  _vflags[fUnknownIds] = flag::Flag("UnknownIds");
  if (_ptype == fOnline) {
    _vflags[fSentRecL1Msm] = flag::Flag("uHTR-L1TMsm");
  }
}

/* virtual */ void TPTask::bookHistograms(DQMStore::IBooker& ib, edm::Run const& r, edm::EventSetup const& es) {
  DQTask::bookHistograms(ib, r, es);

  //	GET WHAT YOU NEED
  edm::ESHandle<HcalDbService> dbs = es.getHandle(hcalDbServiceToken_);
  _emap = dbs->getHcalMapping();
  std::vector<uint32_t> vVME;
  std::vector<uint32_t> vuTCA;
  std::vector<uint32_t> depth0;
  vVME.push_back(HcalElectronicsId(FIBERCH_MIN, FIBER_VME_MIN, SPIGOT_MIN, CRATE_VME_MIN).rawId());
  vuTCA.push_back(HcalElectronicsId(CRATE_uTCA_MIN, SLOT_uTCA_MIN, FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
  _filter_VME.initialize(filter::fFilter, hcaldqm::hashfunctions::fElectronics, vVME);
  _filter_uTCA.initialize(filter::fFilter, hcaldqm::hashfunctions::fElectronics, vuTCA);
  depth0.push_back(HcalTrigTowerDetId(1, 1, 0).rawId());
  _filter_depth0.initialize(filter::fPreserver, hcaldqm::hashfunctions::fTTdepth, depth0);

  //	INITIALIZE FIRST
  //	Et/FG
  _cEtData_TTSubdet.initialize(_name,
                               "EtData",
                               hcaldqm::hashfunctions::fTTSubdet,
                               new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fEt_128),
                               new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),
                               0);
  _cEtEmul_TTSubdet.initialize(_name,
                               "EtEmul",
                               hcaldqm::hashfunctions::fTTSubdet,
                               new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fEt_128),
                               new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),
                               0);
  _cEtCorr_TTSubdet.initialize(_name,
                               "EtCorr_xTS",
                               hcaldqm::hashfunctions::fTTSubdetFW,
                               new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fEtCorr_data),
                               new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fEtCorr_emul),
                               new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),
                               0);
  _cSOIEtCorr_TTSubdet.initialize(_name,
                                  "EtCorr_EmulvsData",
                                  hcaldqm::hashfunctions::fTTSubdetFW,
                                  new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fEtCorr_256),
                                  new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fEtCorr_256),
                                  new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),
                                  0);
  _cSOIEtCorrEmulL1_TTSubdet.initialize(_name,
                                        "EtCorr_EmulvsL1",
                                        hcaldqm::hashfunctions::fTTSubdetFW,
                                        new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fEtCorr_256),
                                        new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fEtCorr_256),
                                        new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),
                                        0);
  for (uint8_t iii = 0; iii < constants::NUM_FGBITS; iii++) {
    _cFGCorr_TTSubdet[iii].initialize(_name,
                                      "FGCorr",
                                      hcaldqm::hashfunctions::fTTSubdet,
                                      new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fFG),
                                      new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fFG),
                                      new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),
                                      0);
  }

  _cEtData_depthlike.initialize(_name,
                                "EtData",
                                new hcaldqm::quantity::TrigTowerQuantity(hcaldqm::quantity::fTTieta),
                                new hcaldqm::quantity::TrigTowerQuantity(hcaldqm::quantity::fTTiphi),
                                new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fEt_256),
                                0);
  _cEtEmul_depthlike.initialize(_name,
                                "EtEmul",
                                new hcaldqm::quantity::TrigTowerQuantity(hcaldqm::quantity::fTTieta),
                                new hcaldqm::quantity::TrigTowerQuantity(hcaldqm::quantity::fTTiphi),
                                new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fEt_256),
                                0);
  _cEtCutData_depthlike.initialize(_name,
                                   "EtCutData",
                                   new hcaldqm::quantity::TrigTowerQuantity(hcaldqm::quantity::fTTieta),
                                   new hcaldqm::quantity::TrigTowerQuantity(hcaldqm::quantity::fTTiphi),
                                   new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fEt_256),
                                   0);
  _cEtCutEmul_depthlike.initialize(_name,
                                   "EtCutEmul",
                                   new hcaldqm::quantity::TrigTowerQuantity(hcaldqm::quantity::fTTieta),
                                   new hcaldqm::quantity::TrigTowerQuantity(hcaldqm::quantity::fTTiphi),
                                   new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fEt_256),
                                   0);

  // Occupancy
  _cOccupancyData_depthlike.initialize(_name,
                                       "OccupancyData",
                                       new hcaldqm::quantity::TrigTowerQuantity(hcaldqm::quantity::fTTieta),
                                       new hcaldqm::quantity::TrigTowerQuantity(hcaldqm::quantity::fTTiphi),
                                       new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),
                                       0);
  _cOccupancyEmul_depthlike.initialize(_name,
                                       "OccupancyEmul",
                                       new hcaldqm::quantity::TrigTowerQuantity(hcaldqm::quantity::fTTieta),
                                       new hcaldqm::quantity::TrigTowerQuantity(hcaldqm::quantity::fTTiphi),
                                       new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),
                                       0);
  _cOccupancyCutData_depthlike.initialize(_name,
                                          "OccupancyCutData",
                                          new hcaldqm::quantity::TrigTowerQuantity(hcaldqm::quantity::fTTieta),
                                          new hcaldqm::quantity::TrigTowerQuantity(hcaldqm::quantity::fTTiphi),
                                          new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),
                                          0);
  _cOccupancyCutEmul_depthlike.initialize(_name,
                                          "OccupancyCutEmul",
                                          new hcaldqm::quantity::TrigTowerQuantity(hcaldqm::quantity::fTTieta),
                                          new hcaldqm::quantity::TrigTowerQuantity(hcaldqm::quantity::fTTiphi),
                                          new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),
                                          0);

  //	Mismatches
  _cEtMsm_depthlike.initialize(_name,
                               "EtMsm",
                               new hcaldqm::quantity::TrigTowerQuantity(hcaldqm::quantity::fTTieta),
                               new hcaldqm::quantity::TrigTowerQuantity(hcaldqm::quantity::fTTiphi),
                               new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                               0);
  _cFGMsm_depthlike.initialize(_name,
                               "FGMsm",
                               new hcaldqm::quantity::TrigTowerQuantity(hcaldqm::quantity::fTTieta),
                               new hcaldqm::quantity::TrigTowerQuantity(hcaldqm::quantity::fTTiphi),
                               new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                               0);

  if (_ptype == fOnline) {
    // Mismatches: sent vs received
    _cEtMsm_uHTR_L1T_depthlike.initialize(_name,
                                          "EtMsm_uHTR_L1T",
                                          new hcaldqm::quantity::TrigTowerQuantity(hcaldqm::quantity::fTTieta),
                                          new hcaldqm::quantity::TrigTowerQuantity(hcaldqm::quantity::fTTiphi),
                                          new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                                          0);
    _cEtMsm_uHTR_L1T_LS.initialize(_name,
                                   "EtMsm_uHTR_L1T_LS",
                                   new hcaldqm::quantity::LumiSection(_maxLS),
                                   new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),
                                   0);
  }

  //	Missing Data w.r.t. Emulator
  _cMsnData_depthlike.initialize(_name,
                                 "MsnData",
                                 new hcaldqm::quantity::TrigTowerQuantity(hcaldqm::quantity::fTTieta),
                                 new hcaldqm::quantity::TrigTowerQuantity(hcaldqm::quantity::fTTiphi),
                                 new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                                 0);
  _cMsnEmul_depthlike.initialize(_name,
                                 "MsnEmul",
                                 new hcaldqm::quantity::TrigTowerQuantity(hcaldqm::quantity::fTTieta),
                                 new hcaldqm::quantity::TrigTowerQuantity(hcaldqm::quantity::fTTiphi),
                                 new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                                 0);

  _cEtCorrRatio_depthlike.initialize(_name,
                                     "EtCorrRatio",
                                     new hcaldqm::quantity::TrigTowerQuantity(hcaldqm::quantity::fTTieta),
                                     new hcaldqm::quantity::TrigTowerQuantity(hcaldqm::quantity::fTTiphi),
                                     new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fRatio_0to2),
                                     0);

  _cOccupancyDatavsBX_TTSubdet.initialize(_name,
                                          "OccupancyDatavsBX",
                                          hcaldqm::hashfunctions::fTTSubdet,
                                          new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fBX),
                                          new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                                          0);
  _cOccupancyEmulvsBX_TTSubdet.initialize(_name,
                                          "OccupancyEmulvsBX",
                                          hcaldqm::hashfunctions::fTTSubdet,
                                          new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fBX),
                                          new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                                          0);
  _cOccupancyCutDatavsBX_TTSubdet.initialize(_name,
                                             "OccupancyCutDatavsBX",
                                             hcaldqm::hashfunctions::fTTSubdet,
                                             new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fBX),
                                             new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                                             0);
  _cOccupancyCutEmulvsBX_TTSubdet.initialize(_name,
                                             "OccupancyCutEmulvsBX",
                                             hcaldqm::hashfunctions::fTTSubdet,
                                             new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fBX),
                                             new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                                             0);

  //	INITIALIZE HISTOGRAMS to be used in Online only!
  if (_ptype == fOnline) {
    _cEtCorr2x3_TTSubdet.initialize(_name,
                                    "EtCorr2x3",
                                    hcaldqm::hashfunctions::fTTSubdet,
                                    new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fEtCorr_256),
                                    new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fEtCorr_256),
                                    new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),
                                    0);
    _cOccupancyData2x3_depthlike.initialize(_name,
                                            "OccupancyData2x3",
                                            new hcaldqm::quantity::TrigTowerQuantity(hcaldqm::quantity::fTTieta2x3),
                                            new hcaldqm::quantity::TrigTowerQuantity(hcaldqm::quantity::fTTiphi),
                                            new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),
                                            0);
    _cOccupancyEmul2x3_depthlike.initialize(_name,
                                            "OccupancyEmul2x3",
                                            new hcaldqm::quantity::TrigTowerQuantity(hcaldqm::quantity::fTTieta2x3),
                                            new hcaldqm::quantity::TrigTowerQuantity(hcaldqm::quantity::fTTiphi),
                                            new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),
                                            0);
    _cEtCutDatavsLS_TTSubdet.initialize(_name,
                                        "EtCutDatavsLS",
                                        hcaldqm::hashfunctions::fTTSubdet,
                                        new hcaldqm::quantity::LumiSection(_maxLS),
                                        new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fEtCorr_256),
                                        0);
    _cEtCutEmulvsLS_TTSubdet.initialize(_name,
                                        "EtCutEmulvsLS",
                                        hcaldqm::hashfunctions::fTTSubdet,
                                        new hcaldqm::quantity::LumiSection(_maxLS),
                                        new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fEtCorr_256),
                                        0);
    _cEtCutDatavsBX_TTSubdet.initialize(_name,
                                        "EtCutDatavsBX",
                                        hcaldqm::hashfunctions::fTTSubdet,
                                        new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fBX),
                                        new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fEtlog2),
                                        new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),
                                        0);
    _cEtCutEmulvsBX_TTSubdet.initialize(_name,
                                        "EtCutEmulvsBX",
                                        hcaldqm::hashfunctions::fTTSubdet,
                                        new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fBX),
                                        new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fEtCorr_256),
                                        0);
    _cEtCorrRatiovsLS_TTSubdet.initialize(_name,
                                          "EtCorrRatiovsLS",
                                          hcaldqm::hashfunctions::fTTSubdet,
                                          new hcaldqm::quantity::LumiSection(_maxLS),
                                          new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fRatio_0to2),
                                          0);
    _cEtCorrRatiovsBX_TTSubdet.initialize(_name,
                                          "EtCorrRatiovsBX",
                                          hcaldqm::hashfunctions::fTTSubdet,
                                          new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fBX),
                                          new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fRatio_0to2),
                                          0);
    _cEtMsmRatiovsLS_TTSubdet.initialize(_name,
                                         "EtMsmRatiovsLS",
                                         hcaldqm::hashfunctions::fTTSubdet,
                                         new hcaldqm::quantity::LumiSection(_maxLS),
                                         new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fRatio),
                                         0);
    _cEtMsmRatiovsBX_TTSubdet.initialize(_name,
                                         "EtMsmRatiovsBX",
                                         hcaldqm::hashfunctions::fTTSubdet,
                                         new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fBX),
                                         new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fRatio),
                                         0);
    _cEtMsmvsLS_TTSubdet.initialize(_name,
                                    "EtMsmvsLS",
                                    hcaldqm::hashfunctions::fTTSubdet,
                                    new hcaldqm::quantity::LumiSection(_maxLS),
                                    new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                                    0);
    _cEtMsmvsBX_TTSubdet.initialize(_name,
                                    "EtMsmvsBX",
                                    hcaldqm::hashfunctions::fTTSubdet,
                                    new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fBX),
                                    new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                                    0);
    _cMsnDatavsLS_TTSubdet.initialize(_name,
                                      "MsnDatavsLS",
                                      hcaldqm::hashfunctions::fTTSubdet,
                                      new hcaldqm::quantity::LumiSection(_maxLS),
                                      new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                                      0);
    _cMsnCutDatavsLS_TTSubdet.initialize(_name,
                                         "MsnCutDatavsLS",
                                         hcaldqm::hashfunctions::fTTSubdet,
                                         new hcaldqm::quantity::LumiSection(_maxLS),
                                         new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                                         0);
    _cMsnDatavsBX_TTSubdet.initialize(_name,
                                      "MsnDatavsBX",
                                      hcaldqm::hashfunctions::fTTSubdet,
                                      new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fBX),
                                      new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                                      0);
    _cMsnCutDatavsBX_TTSubdet.initialize(_name,
                                         "MsnCutDatavsBX",
                                         hcaldqm::hashfunctions::fTTSubdet,
                                         new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fBX),
                                         new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                                         0);
    _cMsnEmulvsLS_TTSubdet.initialize(_name,
                                      "MsnEmulvsLS",
                                      hcaldqm::hashfunctions::fTTSubdet,
                                      new hcaldqm::quantity::LumiSection(_maxLS),
                                      new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                                      0);
    _cMsnCutEmulvsLS_TTSubdet.initialize(_name,
                                         "MsnCutEmulvsLS",
                                         hcaldqm::hashfunctions::fTTSubdet,
                                         new hcaldqm::quantity::LumiSection(_maxLS),
                                         new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                                         0);
    _cMsnEmulvsBX_TTSubdet.initialize(_name,
                                      "MsnEmulvsBX",
                                      hcaldqm::hashfunctions::fTTSubdet,
                                      new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fBX),
                                      new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                                      0);
    _cMsnCutEmulvsBX_TTSubdet.initialize(_name,
                                         "MsnCutEmulvsBX",
                                         hcaldqm::hashfunctions::fTTSubdet,
                                         new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fBX),
                                         new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                                         0);
    _cOccupancyDatavsLS_TTSubdet.initialize(_name,
                                            "OccupancyDatavsLS",
                                            hcaldqm::hashfunctions::fTTSubdet,
                                            new hcaldqm::quantity::LumiSection(_maxLS),
                                            new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                                            0);
    _cOccupancyCutDatavsLS_TTSubdet.initialize(_name,
                                               "OccupancyCutDatavsLS",
                                               hcaldqm::hashfunctions::fTTSubdet,
                                               new hcaldqm::quantity::LumiSection(_maxLS),
                                               new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                                               0);
    _cOccupancyDatavsBX_TTSubdet.initialize(_name,
                                            "OccupancyDatavsBX",
                                            hcaldqm::hashfunctions::fTTSubdet,
                                            new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fBX),
                                            new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                                            0);
    _cOccupancyCutDatavsBX_TTSubdet.initialize(_name,
                                               "OccupancyCutDatavsBX",
                                               hcaldqm::hashfunctions::fTTSubdet,
                                               new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fBX),
                                               new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                                               0);
    _cOccupancyEmulvsLS_TTSubdet.initialize(_name,
                                            "OccupancyEmulvsLS",
                                            hcaldqm::hashfunctions::fTTSubdet,
                                            new hcaldqm::quantity::LumiSection(_maxLS),
                                            new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                                            0);
    _cOccupancyCutEmulvsLS_TTSubdet.initialize(_name,
                                               "OccupancyCutEmulvsLS",
                                               hcaldqm::hashfunctions::fTTSubdet,
                                               new hcaldqm::quantity::LumiSection(_maxLS),
                                               new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                                               0);
    _cOccupancyEmulvsBX_TTSubdet.initialize(_name,
                                            "OccupancyEmulvsBX",
                                            hcaldqm::hashfunctions::fTTSubdet,
                                            new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fBX),
                                            new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                                            0);
    _cOccupancyCutEmulvsBX_TTSubdet.initialize(_name,
                                               "OccupancyCutEmulvsBX",
                                               hcaldqm::hashfunctions::fTTSubdet,
                                               new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fBX),
                                               new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                                               0);

    _cOccupancy_HF_depth.initialize(_name,
                                    "OccupancyDataHF_depth",
                                    new hcaldqm::quantity::TrigTowerQuantity(hcaldqm::quantity::fTTieta),
                                    new hcaldqm::quantity::TrigTowerQuantity(hcaldqm::quantity::fTTiphi),
                                    new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),
                                    0);
    _cOccupancyNoTDC_HF_depth.initialize(_name,
                                         "OccupancyEmulHFNoTDC_depth",
                                         new hcaldqm::quantity::TrigTowerQuantity(hcaldqm::quantity::fTTieta),
                                         new hcaldqm::quantity::TrigTowerQuantity(hcaldqm::quantity::fTTiphi),
                                         new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),
                                         0);
    _cOccupancy_HF_ieta.initialize(_name,
                                   "OccupancyDataHF_ieta",
                                   new hcaldqm::quantity::TrigTowerQuantity(hcaldqm::quantity::fTTieta),
                                   new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),
                                   0),
        _cOccupancyNoTDC_HF_ieta.initialize(_name,
                                            "OccupancyEmulHFNoTDC_ieta",
                                            new hcaldqm::quantity::TrigTowerQuantity(hcaldqm::quantity::fTTieta),
                                            new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),
                                            0);
  }

  // FED-based containers
  if (_ptype != fOffline) {  // hidefed2crate
    std::vector<int> vFEDs = hcaldqm::utilities::getFEDList(_emap);
    std::vector<int> vFEDsVME = hcaldqm::utilities::getFEDVMEList(_emap);
    std::vector<int> vFEDsuTCA = hcaldqm::utilities::getFEDuTCAList(_emap);
    //	push the rawIds of each fed into the vector
    //	this vector is used at endlumi for online state generation
    for (std::vector<int>::const_iterator it = vFEDsVME.begin(); it != vFEDsVME.end(); ++it) {
      _vhashFEDs.push_back(HcalElectronicsId(FIBERCH_MIN, FIBER_VME_MIN, SPIGOT_MIN, (*it) - FED_VME_MIN).rawId());
    }
    for (std::vector<int>::const_iterator it = vFEDsuTCA.begin(); it != vFEDsuTCA.end(); ++it) {
      std::pair<uint16_t, uint16_t> cspair = hcaldqm::utilities::fed2crate(*it);
      _vhashFEDs.push_back(HcalElectronicsId(cspair.first, cspair.second, FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
    }
    _cEtData_ElectronicsuTCA.initialize(_name,
                                        "EtData",
                                        hcaldqm::hashfunctions::fElectronics,
                                        new hcaldqm::quantity::FEDQuantity(vFEDsuTCA),
                                        new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSlotuTCA),
                                        new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fEt_256),
                                        0);
    _cEtEmul_ElectronicsuTCA.initialize(_name,
                                        "EtEmul",
                                        hcaldqm::hashfunctions::fElectronics,
                                        new hcaldqm::quantity::FEDQuantity(vFEDsuTCA),
                                        new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSlotuTCA),
                                        new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fEt_256),
                                        0);
    //	Occupancies
    _cOccupancyData_ElectronicsuTCA.initialize(_name,
                                               "OccupancyData",
                                               hcaldqm::hashfunctions::fElectronics,
                                               new hcaldqm::quantity::FEDQuantity(vFEDsuTCA),
                                               new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSlotuTCA),
                                               new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),
                                               0);
    _cOccupancyEmul_ElectronicsuTCA.initialize(_name,
                                               "OccupancyEmul",
                                               hcaldqm::hashfunctions::fElectronics,
                                               new hcaldqm::quantity::FEDQuantity(vFEDsuTCA),
                                               new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSlotuTCA),
                                               new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),
                                               0);

    _cOccupancyCutData_ElectronicsuTCA.initialize(
        _name,
        "OccupancyCutData",
        hcaldqm::hashfunctions::fElectronics,
        new hcaldqm::quantity::FEDQuantity(vFEDsuTCA),
        new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSlotuTCA),
        new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),
        0);
    _cOccupancyCutEmul_ElectronicsuTCA.initialize(
        _name,
        "OccupancyCutEmul",
        hcaldqm::hashfunctions::fElectronics,
        new hcaldqm::quantity::FEDQuantity(vFEDsuTCA),
        new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSlotuTCA),
        new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),
        0);
    //	Mismatches
    _cEtMsm_ElectronicsuTCA.initialize(_name,
                                       "EtMsm",
                                       hcaldqm::hashfunctions::fElectronics,
                                       new hcaldqm::quantity::FEDQuantity(vFEDsuTCA),
                                       new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSlotuTCA),
                                       new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                                       0);
    _cFGMsm_ElectronicsuTCA.initialize(_name,
                                       "FGMsm",
                                       hcaldqm::hashfunctions::fElectronics,
                                       new hcaldqm::quantity::FEDQuantity(vFEDsuTCA),
                                       new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSlotuTCA),
                                       new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                                       0);

    //	Missing Data w.r.t. Emulator
    _cMsnData_ElectronicsuTCA.initialize(_name,
                                         "MsnData",
                                         hcaldqm::hashfunctions::fElectronics,
                                         new hcaldqm::quantity::FEDQuantity(vFEDsuTCA),
                                         new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSlotuTCA),
                                         new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                                         0);
    _cMsnEmul_ElectronicsuTCA.initialize(_name,
                                         "MsnEmul",
                                         hcaldqm::hashfunctions::fElectronics,
                                         new hcaldqm::quantity::FEDQuantity(vFEDsuTCA),
                                         new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSlotuTCA),
                                         new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                                         0);
    _cEtCorrRatio_ElectronicsuTCA.initialize(_name,
                                             "EtCorrRatio",
                                             hcaldqm::hashfunctions::fElectronics,
                                             new hcaldqm::quantity::FEDQuantity(vFEDsuTCA),
                                             new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSlotuTCA),
                                             new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fRatio_0to2),
                                             0);
    if (_ptype == fOnline) {
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

      _xEtMsm.initialize(hcaldqm::hashfunctions::fFED);
      _xFGMsm.initialize(hcaldqm::hashfunctions::fFED);
      _xNumCorr.initialize(hcaldqm::hashfunctions::fFED);
      _xDataMsn.initialize(hcaldqm::hashfunctions::fFED);
      _xDataTotal.initialize(hcaldqm::hashfunctions::fFED);
      _xEmulMsn.initialize(hcaldqm::hashfunctions::fFED);
      _xEmulTotal.initialize(hcaldqm::hashfunctions::fFED);
      _xSentRecL1Msm.initialize(hcaldqm::hashfunctions::fFED);
    }
  }

  //	BOOK HISTOGRAMS
  char aux[20];
  for (unsigned int iii = 0; iii < constants::NUM_FGBITS; iii++) {
    sprintf(aux, "BIT%d", iii);
    _cFGCorr_TTSubdet[iii].book(ib, _emap, _subsystem, aux);
  }
  _cEtData_TTSubdet.book(ib, _emap, _subsystem);
  _cEtEmul_TTSubdet.book(ib, _emap, _subsystem);
  _cEtCorr_TTSubdet.book(ib, _emap, _subsystem);
  _cSOIEtCorr_TTSubdet.book(ib, _emap, _subsystem);
  _cSOIEtCorrEmulL1_TTSubdet.book(ib, _emap, _subsystem);
  if (_ptype != fOffline) {  // hidefed2crate
    _cEtData_ElectronicsuTCA.book(ib, _emap, _filter_VME, _subsystem);
    _cEtEmul_ElectronicsuTCA.book(ib, _emap, _filter_VME, _subsystem);
  }
  _cEtData_depthlike.book(ib, _subsystem);
  _cEtEmul_depthlike.book(ib, _subsystem);
  _cEtCutData_depthlike.book(ib, _subsystem);
  _cEtCutEmul_depthlike.book(ib, _subsystem);
  if (_ptype != fOffline) {  // hidefed2crate
    _cOccupancyData_ElectronicsuTCA.book(ib, _emap, _filter_VME, _subsystem);
    _cOccupancyEmul_ElectronicsuTCA.book(ib, _emap, _filter_VME, _subsystem);
    _cOccupancyCutData_ElectronicsuTCA.book(ib, _emap, _filter_VME, _subsystem);
    _cOccupancyCutEmul_ElectronicsuTCA.book(ib, _emap, _filter_VME, _subsystem);
  }
  _cOccupancyData_depthlike.book(ib, _subsystem);
  _cOccupancyEmul_depthlike.book(ib, _subsystem);
  _cOccupancyCutData_depthlike.book(ib, _subsystem);
  _cOccupancyCutEmul_depthlike.book(ib, _subsystem);

  _cEtCorrRatio_depthlike.book(ib, _subsystem);
  _cEtMsm_depthlike.book(ib, _subsystem);
  _cFGMsm_depthlike.book(ib, _subsystem);
  _cMsnData_depthlike.book(ib, _subsystem);
  _cMsnEmul_depthlike.book(ib, _subsystem);

  if (_ptype == fOnline) {
    _cEtMsm_uHTR_L1T_depthlike.book(ib, _subsystem);
    _cEtMsm_uHTR_L1T_LS.book(ib, _subsystem);
  }

  if (_ptype != fOffline) {  // hidefed2crate
    _cEtMsm_ElectronicsuTCA.book(ib, _emap, _filter_VME, _subsystem);
    _cFGMsm_ElectronicsuTCA.book(ib, _emap, _filter_VME, _subsystem);
    _cMsnData_ElectronicsuTCA.book(ib, _emap, _filter_VME, _subsystem);
    _cMsnEmul_ElectronicsuTCA.book(ib, _emap, _filter_VME, _subsystem);
    _cEtCorrRatio_ElectronicsuTCA.book(ib, _emap, _filter_VME, _subsystem);
  }

  //	whatever has to go online only goes here
  if (_ptype == fOnline) {
    _cEtCorr2x3_TTSubdet.book(ib, _emap, _subsystem);
    _cOccupancyData2x3_depthlike.book(ib, _subsystem);
    _cOccupancyEmul2x3_depthlike.book(ib, _subsystem);
    _cEtCutDatavsLS_TTSubdet.book(ib, _emap, _subsystem);
    _cEtCutEmulvsLS_TTSubdet.book(ib, _emap, _subsystem);
    _cEtCutDatavsBX_TTSubdet.book(ib, _emap, _subsystem);
    _cEtCutEmulvsBX_TTSubdet.book(ib, _emap, _subsystem);
    _cEtCorrRatiovsLS_TTSubdet.book(ib, _emap, _subsystem);
    _cEtCorrRatiovsBX_TTSubdet.book(ib, _emap, _subsystem);
    _cEtMsmvsLS_TTSubdet.book(ib, _emap, _subsystem);
    _cEtMsmvsBX_TTSubdet.book(ib, _emap, _subsystem);
    _cEtMsmRatiovsLS_TTSubdet.book(ib, _emap, _subsystem);
    _cEtMsmRatiovsBX_TTSubdet.book(ib, _emap, _subsystem);
    _cMsnDatavsLS_TTSubdet.book(ib, _emap, _subsystem);
    _cMsnCutDatavsLS_TTSubdet.book(ib, _emap, _subsystem);
    _cMsnDatavsBX_TTSubdet.book(ib, _emap, _subsystem);
    _cMsnCutDatavsBX_TTSubdet.book(ib, _emap, _subsystem);
    _cMsnEmulvsLS_TTSubdet.book(ib, _emap, _subsystem);
    _cMsnCutEmulvsLS_TTSubdet.book(ib, _emap, _subsystem);
    _cMsnEmulvsBX_TTSubdet.book(ib, _emap, _subsystem);
    _cMsnCutEmulvsBX_TTSubdet.book(ib, _emap, _subsystem);
    _cOccupancyDatavsBX_TTSubdet.book(ib, _emap, _subsystem);
    _cOccupancyEmulvsBX_TTSubdet.book(ib, _emap, _subsystem);
    _cOccupancyCutDatavsBX_TTSubdet.book(ib, _emap, _subsystem);
    _cOccupancyCutEmulvsBX_TTSubdet.book(ib, _emap, _subsystem);
    _cOccupancyDatavsLS_TTSubdet.book(ib, _emap, _subsystem);
    _cOccupancyEmulvsLS_TTSubdet.book(ib, _emap, _subsystem);
    _cOccupancyCutDatavsLS_TTSubdet.book(ib, _emap, _subsystem);
    _cOccupancyCutEmulvsLS_TTSubdet.book(ib, _emap, _subsystem);
    _cSummaryvsLS_FED.book(ib, _emap, _subsystem);
    _cSummaryvsLS.book(ib, _subsystem);

    _xEtMsm.book(_emap);
    _xFGMsm.book(_emap);
    _xNumCorr.book(_emap);
    _xDataMsn.book(_emap);
    _xDataTotal.book(_emap);
    _xEmulMsn.book(_emap);
    _xEmulTotal.book(_emap);
    _xSentRecL1Msm.book(_emap);

    _cOccupancy_HF_depth.book(ib, _subsystem);
    _cOccupancyNoTDC_HF_depth.book(ib, _subsystem);
    _cOccupancy_HF_ieta.book(ib, _subsystem);
    _cOccupancyNoTDC_HF_ieta.book(ib, _subsystem);
  }

  //	initialize the hash map
  _ehashmap.initialize(_emap, hcaldqm::electronicsmap::fT2EHashMap);

  //	book the flag for unknown ids and the online guy as well
  ib.setCurrentFolder(_subsystem + "/" + _name);
  auto scope = DQMStore::IBooker::UseLumiScope(ib);
  meUnknownIds1LS = ib.book1DD("UnknownIds", "UnknownIds", 1, 0, 1);
  _unknownIdsPresent = false;
}

/* virtual */ void TPTask::_resetMonitors(hcaldqm::UpdateFreq uf) {
  DQTask::_resetMonitors(uf);
  switch (uf) {
    case hcaldqm::f1LS:
      _unknownIdsPresent = false;
      break;
    default:
      break;
  }
}

/* virtual */ void TPTask::_process(edm::Event const& e, edm::EventSetup const&) {
  edm::Handle<HcalTrigPrimDigiCollection> cdata;
  edm::Handle<HcalTrigPrimDigiCollection> cdataL1Rec;
  edm::Handle<HcalTrigPrimDigiCollection> cemul;
  edm::Handle<HcalTrigPrimDigiCollection> cemul_noTDCCut;
  if (!e.getByToken(_tokData, cdata))
    _logger.dqmthrow("Collection HcalTrigPrimDigiCollection isn't available: " + _tagData.label() + " " +
                     _tagData.instance());
  if (_ptype == fOnline) {
    if (!e.getByToken(_tokDataL1Rec, cdataL1Rec))
      _logger.dqmthrow("Collection HcalTrigPrimDigiCollection isn't available: " + _tagDataL1Rec.label() + " " +
                       _tagDataL1Rec.instance());
  }
  if (!e.getByToken(_tokEmul, cemul))
    _logger.dqmthrow("Collection HcalTrigPrimDigiCollection isn't available: " + _tagEmul.label() + " " +
                     _tagEmul.instance());
  if (_ptype == fOnline) {
    if (!e.getByToken(_tokEmulNoTDCCut, cemul_noTDCCut)) {
      _logger.dqmthrow("Collection HcalTrigPrimDigiCollection isn't available: " + _tagEmulNoTDCCut.label() + " " +
                       _tagEmulNoTDCCut.instance());
    }
  }

  //	extract some info per event
  int bx = e.bunchCrossing();

  auto lumiCache = luminosityBlockCache(e.getLuminosityBlock().index());
  _currentLS = lumiCache->currentLS;

  //	some summaries... per event
  int numHBHE(0), numHF(0), numCutHBHE(0), numCutHF(0);
  int numCorrHBHE(0), numCorrHF(0);
  int numMsmHBHE(0), numMsmHF(0);
  int numMsnHBHE(0), numMsnHF(0), numMsnCutHBHE(0), numMsnCutHF(0);

  //	for explanation see RecHit or Digi Tasks
  uint32_t rawidHBHEValid = 0;
  uint32_t rawidHFValid = 0;

  /*
	 * STEP1: 
	 * Loop over the data digis and 
	 * - do ... for all the data digis
	 * - find the emulator digi
	 * --- compare soi Et
	 * --- compare soi FG
	 * --- Do not fill anything for emulator Et!!!
	 */
  for (HcalTrigPrimDigiCollection::const_iterator it = cdata->begin(); it != cdata->end(); ++it) {
    //	Explicit check on the DetIds present in the Collection
    HcalTrigTowerDetId tid = it->id();
    uint32_t rawid = _ehashmap.lookup(tid);
    if (rawid == 0) {
      meUnknownIds1LS->Fill(1);
      _unknownIdsPresent = true;
      continue;
    }
    HcalElectronicsId const& eid(rawid);
    if (tid.ietaAbs() >= 29)
      rawidHFValid = tid.rawId();
    else
      rawidHBHEValid = tid.rawId();

    //
    //	HF 2x3 TPs Treat theam separately and only for ONLINE!
    //
    if (tid.version() == 0 && tid.ietaAbs() >= 29) {
      //	do this only for online processing
      if (_ptype == fOnline) {
        _cOccupancyData2x3_depthlike.fill(tid);
        HcalTrigPrimDigiCollection::const_iterator jt = cemul->find(tid);
        if (jt != cemul->end())
          _cEtCorr2x3_TTSubdet.fill(tid, it->SOI_compressedEt(), jt->SOI_compressedEt());
      }

      //	skip to the next tp digi
      continue;
    }

    //	FROM THIS POINT, HBHE + 1x1 HF TPs
    int soiEt_d = it->t0().compressedEt();
    int soiFG_d[constants::NUM_FGBITS];
    for (uint32_t ibit = 0; ibit < constants::NUM_FGBITS; ibit++)
      soiFG_d[ibit] = it->t0().fineGrain(ibit) ? 1 : 0;
    tid.ietaAbs() >= 29 ? numHF++ : numHBHE++;

    //	 fill w/o a cut
    _cEtData_TTSubdet.fill(tid, soiEt_d);
    _cEtData_depthlike.fill(tid, soiEt_d);
    _cOccupancyData_depthlike.fill(tid);

    if (_ptype == fOnline) {
      if (tid.ietaAbs() >= 29) {
        if (soiEt_d > 0) {
          _cOccupancy_HF_depth.fill(tid);
          _cOccupancy_HF_ieta.fill(tid);
        }
      }
    }
    if (_ptype != fOffline) {  // hidefed2crate
      if (!eid.isVMEid()) {
        _cOccupancyData_ElectronicsuTCA.fill(eid);
        _cEtData_ElectronicsuTCA.fill(eid, soiEt_d);
      }
    }

    //	FILL w/a CUT
    if (soiEt_d > _cutEt) {
      tid.ietaAbs() >= 29 ? numCutHF++ : numCutHBHE++;
      _cOccupancyCutData_depthlike.fill(tid);
      _cEtCutData_depthlike.fill(tid, soiEt_d);

      //	ONLINE ONLY!
      if (_ptype == fOnline) {
        _cEtCutDatavsLS_TTSubdet.fill(tid, _currentLS, soiEt_d);
        _cEtCutDatavsBX_TTSubdet.fill(tid, bx, log2(soiEt_d + 1.));
        _xDataTotal.get(eid)++;
      }
      //	^^^ONLINE ONLY!
      if (_ptype != fOffline) {  // hidefed2crate
        if (!eid.isVMEid())
          _cOccupancyCutData_ElectronicsuTCA.fill(eid);
      }
    }

    //	FIND the EMULATOR DIGI
    HcalTrigPrimDigiCollection::const_iterator jt = cemul->find(tid);
    if (jt != cemul->end()) {
      //	if PRESENT!
      int soiEt_e = jt->SOI_compressedEt();
      int soiFG_e[constants::NUM_FGBITS];
      for (uint32_t ibit = 0; ibit < constants::NUM_FGBITS; ibit++)
        soiFG_e[ibit] = jt->t0().fineGrain(ibit) ? 1 : 0;
      //	if both are zeroes => set 1
      double rEt =
          soiEt_d == 0 && soiEt_e == 0 ? 1 : double(std::min(soiEt_d, soiEt_e)) / double(std::max(soiEt_e, soiEt_d));

      //	ONLINE ONLY!
      if (_ptype == fOnline) {
        _xNumCorr.get(eid)++;
        tid.ietaAbs() >= 29 ? numCorrHF++ : numCorrHBHE++;
        _cEtCorrRatiovsLS_TTSubdet.fill(tid, _currentLS, rEt);
        _cEtCorrRatiovsBX_TTSubdet.fill(tid, bx, rEt);
      }
      //	^^^ONLINE ONLY!

      _cEtCorrRatio_depthlike.fill(tid, rEt);
      _cSOIEtCorr_TTSubdet.fill(tid, eid, soiEt_d, soiEt_e);
      for (int ci = 0; ci < 4; ci++) {
        for (int cj = 0; cj < 4; cj++) {
          if (ci < it->size() && cj < jt->size()) {
            if ((ci == cj) || (it->sample(ci).compressedEt() > 0 && jt->sample(cj).compressedEt() > 0)) {
              _cEtCorr_TTSubdet.fill(
                  tid, eid, 256 * ci + it->sample(ci).compressedEt(), 256 * cj + jt->sample(cj).compressedEt());
            }
          }
        }
      }
      for (uint32_t ibit = 0; ibit < constants::NUM_FGBITS; ibit++)
        _cFGCorr_TTSubdet[ibit].fill(tid, soiFG_d[ibit], soiFG_e[ibit]);
      //	FILL w/o a CUT
      if (_ptype != fOffline) {  // hidefed2crate
        if (!eid.isVMEid())
          _cEtCorrRatio_ElectronicsuTCA.fill(eid, rEt);
      }

      //	if SOI Et are not equal
      //	fill mismatched
      if (soiEt_d != soiEt_e) {
        tid.ietaAbs() >= 29 ? numMsmHF++ : numMsmHBHE++;
        _cEtMsm_depthlike.fill(tid);
        if (_ptype != fOffline) {  // hidefed2crate
          if (!eid.isVMEid())
            _cEtMsm_ElectronicsuTCA.fill(eid);
        }
        if (_ptype == fOnline)
          _xEtMsm.get(eid)++;
      }
      //	 if SOI FG are not equal
      //	 fill mismatched.
      //	 Do this comparison only for FG Bits that are commissioned
      for (uint32_t ibit = 0; ibit < constants::NUM_FGBITS; ibit++)
        if (soiFG_d[ibit] != soiFG_e[ibit] && _vFGBitsReady[ibit]) {
          _cFGMsm_depthlike.fill(tid);
          if (_ptype != fOffline) {  // hidefed2crate
            if (!eid.isVMEid())
              _cFGMsm_ElectronicsuTCA.fill(eid);
          }
          if (_ptype == fOnline)
            _xFGMsm.get(eid)++;
        }
    } else {
      //	IF MISSING
      _cEtCorr_TTSubdet.fill(tid, eid, soiEt_d, -1);
      _cSOIEtCorr_TTSubdet.fill(tid, eid, soiEt_d, -1);
      _cMsnEmul_depthlike.fill(tid);
      tid.ietaAbs() >= 29 ? numMsnHF++ : numMsnHBHE++;
      if (_ptype != fOffline) {  // hidefed2crate
        if (!eid.isVMEid())
          _cMsnEmul_ElectronicsuTCA.fill(eid);
      }

      if (soiEt_d > _cutEt) {
        tid.ietaAbs() >= 29 ? numMsnCutHF++ : numMsnCutHBHE++;
        if (_ptype == fOnline)
          _xEmulMsn.get(eid)++;
      }
    }
  }

  if (_ptype == fOnline) {
    for (HcalTrigPrimDigiCollection::const_iterator it = cemul_noTDCCut->begin(); it != cemul_noTDCCut->end(); ++it) {
      //	Explicit check on the DetIds present in the Collection
      HcalTrigTowerDetId tid = it->id();
      uint32_t rawid = _ehashmap.lookup(tid);
      if (rawid == 0) {
        continue;
      }
      if (tid.version() == 0 && tid.ietaAbs() >= 29) {
        continue;
      }
      int soiEt_e = it->SOI_compressedEt();
      if (tid.ietaAbs() >= 29) {
        if (soiEt_e > 0) {
          _cOccupancyNoTDC_HF_depth.fill(tid);
          _cOccupancyNoTDC_HF_ieta.fill(tid);
        }
      }
    }
  }

  if (rawidHFValid != 0 && rawidHBHEValid != 0) {
    //	ONLINE ONLY!
    if (_ptype == fOnline) {
      _cOccupancyDatavsBX_TTSubdet.fill(HcalTrigTowerDetId(rawidHBHEValid), bx, numHBHE);
      _cOccupancyDatavsBX_TTSubdet.fill(HcalTrigTowerDetId(rawidHFValid), bx, numHF);
      _cOccupancyCutDatavsBX_TTSubdet.fill(HcalTrigTowerDetId(rawidHBHEValid), bx, numCutHBHE);
      _cOccupancyCutDatavsBX_TTSubdet.fill(HcalTrigTowerDetId(rawidHFValid), bx, numCutHF);
      _cOccupancyDatavsLS_TTSubdet.fill(HcalTrigTowerDetId(rawidHBHEValid), _currentLS, numHBHE);
      _cOccupancyDatavsLS_TTSubdet.fill(HcalTrigTowerDetId(rawidHFValid), _currentLS, numHF);
      _cOccupancyCutDatavsLS_TTSubdet.fill(HcalTrigTowerDetId(rawidHBHEValid), _currentLS, numCutHBHE);
      _cOccupancyCutDatavsLS_TTSubdet.fill(HcalTrigTowerDetId(rawidHFValid), _currentLS, numCutHF);

      _cEtMsmvsLS_TTSubdet.fill(HcalTrigTowerDetId(rawidHBHEValid), _currentLS, numMsmHBHE);
      _cEtMsmvsLS_TTSubdet.fill(HcalTrigTowerDetId(rawidHFValid), _currentLS, numMsmHF);
      _cEtMsmvsBX_TTSubdet.fill(HcalTrigTowerDetId(rawidHBHEValid), bx, numMsmHBHE);
      _cEtMsmvsBX_TTSubdet.fill(HcalTrigTowerDetId(rawidHFValid), bx, numMsmHF);

      _cEtMsmRatiovsLS_TTSubdet.fill(
          HcalTrigTowerDetId(rawidHBHEValid), _currentLS, double(numMsmHBHE) / double(numCorrHBHE));
      _cEtMsmRatiovsLS_TTSubdet.fill(
          HcalTrigTowerDetId(rawidHFValid), _currentLS, double(numMsmHF) / double(numCorrHF));
      _cEtMsmRatiovsBX_TTSubdet.fill(HcalTrigTowerDetId(rawidHBHEValid), bx, double(numMsmHBHE) / double(numCorrHBHE));
      _cEtMsmRatiovsBX_TTSubdet.fill(HcalTrigTowerDetId(rawidHFValid), bx, double(numMsmHF) / double(numCorrHF));

      _cMsnEmulvsLS_TTSubdet.fill(HcalTrigTowerDetId(rawidHBHEValid), _currentLS, numMsnHBHE);
      _cMsnEmulvsLS_TTSubdet.fill(HcalTrigTowerDetId(rawidHFValid), _currentLS, numMsnHF);
      _cMsnCutEmulvsLS_TTSubdet.fill(HcalTrigTowerDetId(rawidHBHEValid), _currentLS, numMsnCutHBHE);
      _cMsnCutEmulvsLS_TTSubdet.fill(HcalTrigTowerDetId(rawidHFValid), _currentLS, numMsnCutHF);

      _cMsnEmulvsBX_TTSubdet.fill(HcalTrigTowerDetId(rawidHBHEValid), bx, numMsnHBHE);
      _cMsnEmulvsBX_TTSubdet.fill(HcalTrigTowerDetId(rawidHFValid), bx, numMsnHF);
      _cMsnCutEmulvsBX_TTSubdet.fill(HcalTrigTowerDetId(rawidHBHEValid), bx, numMsnCutHBHE);
      _cMsnCutEmulvsBX_TTSubdet.fill(HcalTrigTowerDetId(rawidHFValid), bx, numMsnCutHF);
    }
  }

  numHBHE = 0;
  numHF = 0;
  numMsnHBHE = 0;
  numMsnHF = 0;
  numCutHBHE = 0;
  numCutHF = 0;

  //	reset
  rawidHBHEValid = 0;
  rawidHFValid = 0;

  /*
	 *	STEP2:
	 *	Loop over the emulator digis and 
	 *	- do ... for all the emulator digis
	 *	- find data digi and 
	 *	--- if found skip
	 *	--- if not found - fill the missing Data plot
	 */
  for (HcalTrigPrimDigiCollection::const_iterator it = cemul->begin(); it != cemul->end(); ++it) {
    //	Explicit check on the DetIds present in the Collection
    HcalTrigTowerDetId tid = it->id();
    uint32_t rawid = _ehashmap.lookup(tid);
    if (rawid == 0) {
      meUnknownIds1LS->Fill(1);
      _unknownIdsPresent = true;
      continue;
    }
    HcalElectronicsId const& eid(rawid);
    if (tid.ietaAbs() >= 29)
      rawidHFValid = tid.rawId();
    else
      rawidHBHEValid = tid.rawId();

    //	HF 2x3 TPs. Only do it for Online!!!
    if (tid.version() == 0 && tid.ietaAbs() >= 29) {
      //	only do this for online processing
      if (_ptype == fOnline)
        _cOccupancyEmul2x3_depthlike.fill(tid);
      continue;
    }
    int soiEt = it->SOI_compressedEt();

    //	FILL/INCREMENT w/o a CUT
    tid.ietaAbs() >= 29 ? numHF++ : numHBHE++;
    _cEtEmul_TTSubdet.fill(tid, soiEt);
    _cEtEmul_depthlike.fill(tid, soiEt);
    _cOccupancyEmul_depthlike.fill(tid);
    if (_ptype != fOffline) {  // hidefed2crate
      if (!eid.isVMEid()) {
        _cOccupancyEmul_ElectronicsuTCA.fill(eid);
        _cEtEmul_ElectronicsuTCA.fill(eid, soiEt);
      }
    }

    //	FILL w/ a CUT
    if (soiEt > _cutEt) {
      tid.ietaAbs() >= 29 ? numCutHF++ : numCutHBHE++;
      _cOccupancyCutEmul_depthlike.fill(tid);
      _cEtCutEmul_depthlike.fill(tid, soiEt);
      if (_ptype != fOffline) {  // hidefed2crate
        if (!eid.isVMEid())
          _cOccupancyCutEmul_ElectronicsuTCA.fill(eid);
      }

      //	ONLINE ONLY!
      if (_ptype == fOnline) {
        _cEtCutEmulvsLS_TTSubdet.fill(tid, _currentLS, soiEt);
        _cEtCutEmulvsBX_TTSubdet.fill(tid, bx, soiEt);
        _xEmulTotal.get(eid)++;
      }
      //	^^^ONLINE ONLY!
    }

    // Look for a data digi.
    // Do not perform if the emulated digi is zero suppressed.
    if (!(it->zsMarkAndPass())) {
      HcalTrigPrimDigiCollection::const_iterator jt = cdata->find(tid);
      if (jt == cdata->end()) {
        tid.ietaAbs() >= 29 ? numMsnHF++ : numMsnHBHE++;
        _cEtCorr_TTSubdet.fill(tid, eid, -1, soiEt);
        _cSOIEtCorr_TTSubdet.fill(tid, eid, -1, soiEt);
        _cMsnData_depthlike.fill(tid);
        if (_ptype != fOffline) {  // hidefed2crate
          if (!eid.isVMEid())
            _cMsnData_ElectronicsuTCA.fill(eid);
        }
        if (soiEt > _cutEt) {
          tid.ietaAbs() >= 29 ? numMsnCutHF++ : numMsnCutHBHE++;
          if (_ptype == fOnline)
            _xDataMsn.get(eid)++;
        }
      }
    }
  }

  //	ONLINE ONLY!
  if (_ptype == fOnline) {
    if (rawidHBHEValid != 0) {
      _cOccupancyEmulvsBX_TTSubdet.fill(HcalTrigTowerDetId(rawidHBHEValid), bx, numHBHE);
      _cOccupancyCutEmulvsBX_TTSubdet.fill(HcalTrigTowerDetId(rawidHBHEValid), bx, numCutHBHE);
      _cOccupancyEmulvsLS_TTSubdet.fill(HcalTrigTowerDetId(rawidHBHEValid), _currentLS, numHBHE);
      _cOccupancyCutEmulvsLS_TTSubdet.fill(HcalTrigTowerDetId(rawidHBHEValid), _currentLS, numCutHBHE);
      _cMsnDatavsLS_TTSubdet.fill(HcalTrigTowerDetId(rawidHBHEValid), _currentLS, numMsnHBHE);
      _cMsnCutDatavsLS_TTSubdet.fill(HcalTrigTowerDetId(rawidHBHEValid), _currentLS, numMsnCutHBHE);
      _cMsnDatavsBX_TTSubdet.fill(HcalTrigTowerDetId(rawidHBHEValid), bx, numMsnHBHE);
      _cMsnCutDatavsBX_TTSubdet.fill(HcalTrigTowerDetId(rawidHBHEValid), bx, numMsnCutHBHE);
    }
    if (rawidHFValid != 0) {
      _cOccupancyEmulvsBX_TTSubdet.fill(HcalTrigTowerDetId(rawidHFValid), bx, numHF);
      _cOccupancyCutEmulvsBX_TTSubdet.fill(HcalTrigTowerDetId(rawidHFValid), bx, numCutHF);
      _cOccupancyEmulvsLS_TTSubdet.fill(HcalTrigTowerDetId(rawidHFValid), _currentLS, numHF);
      _cOccupancyCutEmulvsLS_TTSubdet.fill(HcalTrigTowerDetId(rawidHFValid), _currentLS, numCutHF);
      _cMsnDatavsLS_TTSubdet.fill(HcalTrigTowerDetId(rawidHFValid), _currentLS, numMsnHF);
      _cMsnCutDatavsLS_TTSubdet.fill(HcalTrigTowerDetId(rawidHFValid), _currentLS, numMsnCutHF);
      _cMsnDatavsBX_TTSubdet.fill(HcalTrigTowerDetId(rawidHFValid), bx, numMsnHF);
      _cMsnCutDatavsBX_TTSubdet.fill(HcalTrigTowerDetId(rawidHFValid), bx, numMsnCutHF);
    }
    //	^^^ONLINE ONLY!
  }

  if (_ptype == fOnline) {
    // Compare the sent ("uHTR") and received (L1T "layer1") TPs
    // This algorithm is copied from DQM/L1TMonitor/src/L1TStage2CaloLayer1.cc
    // ...but it turns out to be extremely useful for detecting uHTR problems

    _vEmulTPDigis_SentRec.clear();
    ComparisonHelper::zip(cemul->begin(),
                          cemul->end(),
                          cdataL1Rec->begin(),
                          cdataL1Rec->end(),
                          std::inserter(_vEmulTPDigis_SentRec, _vEmulTPDigis_SentRec.begin()),
                          HcalTrigPrimDigiCollection::key_compare());

    // comparison between emulation TP and L1 TP
    for (const auto& tpPair : _vEmulTPDigis_SentRec) {
      const auto& sentTp = tpPair.first;
      const auto& recdTp = tpPair.second;
      const int ieta = sentTp.id().ieta();
      if (abs(ieta) > 28 && sentTp.id().version() != 1)
        continue;

      const bool towerMasked = recdTp.sample(0).raw() & (1 << 13);
      const bool linkError = recdTp.sample(0).raw() & (1 << 15);
      if (towerMasked || linkError)
        continue;

      HcalTrigTowerDetId tid = sentTp.id();
      uint32_t rawid = _ehashmap.lookup(tid);
      if (rawid == 0) {
        continue;
      }
      HcalElectronicsId const& eid(rawid);

      _cSOIEtCorrEmulL1_TTSubdet.fill(tid, eid, recdTp.SOI_compressedEt(), sentTp.SOI_compressedEt());
    }

    // comparison between sent data TP and L1 TP
    _vTPDigis_SentRec.clear();
    ComparisonHelper::zip(cdata->begin(),
                          cdata->end(),
                          cdataL1Rec->begin(),
                          cdataL1Rec->end(),
                          std::inserter(_vTPDigis_SentRec, _vTPDigis_SentRec.begin()),
                          HcalTrigPrimDigiCollection::key_compare());

    for (const auto& tpPair : _vTPDigis_SentRec) {
      // From here, literal copy pasta from L1T
      const auto& sentTp = tpPair.first;
      const auto& recdTp = tpPair.second;
      const int ieta = sentTp.id().ieta();
      if (abs(ieta) > 28 && sentTp.id().version() != 1)
        continue;
      //const int iphi = sentTp.id().iphi();
      const bool towerMasked = recdTp.sample(0).raw() & (1 << 13);
      //const bool linkMasked  = recdTp.sample(0).raw() & (1<<14);
      const bool linkError = recdTp.sample(0).raw() & (1 << 15);

      if (towerMasked || linkError) {
        // Do not compare if known to be bad
        continue;
      }

      HcalTrigTowerDetId tid = sentTp.id();
      uint32_t rawid = _ehashmap.lookup(tid);
      if (rawid == 0) {
        continue;
      }
      HcalElectronicsId const& eid(rawid);

      const bool HetAgreement = sentTp.SOI_compressedEt() == recdTp.SOI_compressedEt();
      const bool Hfb1Agreement = sentTp.SOI_fineGrain() == recdTp.SOI_fineGrain();
      // Ignore minBias (FB2) bit if we receieve 0 ET, which means it is likely zero-suppressed on HCal readout side
      const bool Hfb2Agreement =
          (abs(ieta) < 29) ? true
                           : (recdTp.SOI_compressedEt() == 0 || (sentTp.SOI_fineGrain(1) == recdTp.SOI_fineGrain(1)));
      if (!(HetAgreement && Hfb1Agreement && Hfb2Agreement)) {
        _cEtMsm_uHTR_L1T_depthlike.fill(tid);
        _cEtMsm_uHTR_L1T_LS.fill(_currentLS);
        _xSentRecL1Msm.get(eid)++;
      }
    }
  }
}

std::shared_ptr<hcaldqm::Cache> TPTask::globalBeginLuminosityBlock(edm::LuminosityBlock const& lb,
                                                                   edm::EventSetup const& es) const {
  return DQTask::globalBeginLuminosityBlock(lb, es);
}

/* virtual */ void TPTask::globalEndLuminosityBlock(edm::LuminosityBlock const& lb, edm::EventSetup const& es) {
  if (_ptype != fOnline)
    return;

  auto lumiCache = luminosityBlockCache(lb.index());
  _currentLS = lumiCache->currentLS;

  //
  //	GENERATE STATUS ONLY FOR ONLINE!
  //
  for (std::vector<uint32_t>::const_iterator it = _vhashFEDs.begin(); it != _vhashFEDs.end(); ++it) {
    flag::Flag fSum("TP");
    HcalElectronicsId eid = HcalElectronicsId(*it);

    std::vector<uint32_t>::const_iterator cit = std::find(_vcdaqEids.begin(), _vcdaqEids.end(), *it);
    if (cit == _vcdaqEids.end()) {
      //	not @cDAQ
      for (uint32_t iflag = 0; iflag < _vflags.size(); iflag++)
        _cSummaryvsLS_FED.setBinContent(eid, _currentLS, int(iflag), int(flag::fNCDAQ));
      _cSummaryvsLS.setBinContent(eid, _currentLS, int(flag::fNCDAQ));
      continue;
    }

    if (hcaldqm::utilities::isFEDHBHE(eid) || hcaldqm::utilities::isFEDHF(eid)) {
      //	FED is @cDAQ
      double etmsm = _xNumCorr.get(eid) > 0 ? double(_xEtMsm.get(eid)) / double(_xNumCorr.get(eid)) : 0;
      /*	
			 * UNUSED VARS
			 * double dmsm = _xDataTotal.get(eid)>0?
				double(_xDataMsn.get(eid))/double(_xDataTotal.get(eid)):0;
			double emsm = _xEmulTotal.get(eid)>0?
				double(_xEmulMsn.get(eid))/double(_xEmulTotal.get(eid)):0;
			double fgmsm = _xNumCorr.get(eid)>0?
				double(_xFGMsm.get(eid))/double(_xNumCorr.get(eid)):0;				
				*/
      if (etmsm >= _thresh_EtMsmRate_high)
        _vflags[fEtMsm]._state = flag::fBAD;
      else if (etmsm >= _thresh_EtMsmRate_low)
        _vflags[fEtMsm]._state = flag::fPROBLEMATIC;
      else
        _vflags[fEtMsm]._state = flag::fGOOD;
      /*
			 *	DISABLE THESE FLAGS FOR ONLINE FOR NOW!
			if (dmsm>=_thresh_DataMsn)
				_vflags[fDataMsn]._state = flag::fBAD;
			else
				_vflags[fDataMsn]._state = flag::fGOOD;
			if (emsm>=_thresh_EmulMsn)
				_vflags[fEmulMsn]._state = flag::fBAD;
			else
				_vflags[fEmulMsn]._state = flag::fGOOD;
				*/

      if (_ptype == fOnline) {
        if (_xSentRecL1Msm.get(eid) >= 1) {
          _vflags[fSentRecL1Msm]._state = flag::fBAD;
        } else {
          _vflags[fSentRecL1Msm]._state = flag::fGOOD;
        }
      }
    }

    if (_unknownIdsPresent)
      _vflags[fUnknownIds]._state = flag::fBAD;
    else
      _vflags[fUnknownIds]._state = flag::fGOOD;

    int iflag = 0;
    for (std::vector<flag::Flag>::iterator ft = _vflags.begin(); ft != _vflags.end(); ++ft) {
      _cSummaryvsLS_FED.setBinContent(eid, _currentLS, int(iflag), ft->_state);
      fSum += (*ft);
      iflag++;

      //	this is the MUST!
      //	reset after using this flag
      ft->reset();
    }
    _cSummaryvsLS.setBinContent(eid, _currentLS, int(fSum._state));
  }

  //	reset...
  _xEtMsm.reset();
  _xFGMsm.reset();
  _xNumCorr.reset();
  _xDataMsn.reset();
  _xDataTotal.reset();
  _xEmulMsn.reset();
  _xEmulTotal.reset();

  //	in the end always do the DQTask::endLumi
  DQTask::globalEndLuminosityBlock(lb, es);
}

DEFINE_FWK_MODULE(TPTask);
