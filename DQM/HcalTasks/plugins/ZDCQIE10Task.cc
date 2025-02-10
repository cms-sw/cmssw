#include "DQM/HcalTasks/interface/ZDCQIE10Task.h"
using namespace std;
using namespace hcaldqm;
using namespace hcaldqm::constants;
using namespace hcaldqm::filter;

ZDCQIE10Task::ZDCQIE10Task(edm::ParameterSet const& ps)

    : DQTask(ps), hcalDbServiceToken_(esConsumes<HcalDbService, HcalDbRecord, edm::Transition::BeginRun>()) {
  //tags
  _tagQIE10 = ps.getUntrackedParameter<edm::InputTag>("tagQIE10", edm::InputTag("hcalDigis", "ZDC"));
  _tokQIE10 = consumes<QIE10DigiCollection>(_tagQIE10);

  sumTag = ps.getUntrackedParameter<edm::InputTag>("etSumTag", edm::InputTag("etSumZdcProducer", ""));
  sumToken_ = consumes<l1t::EtSumBxCollection>(sumTag);

  htopoToken_ = esConsumes<HcalTopology, HcalRecNumberingRecord>();
  paramsToken_ = esConsumes<HcalLongRecoParams, HcalLongRecoParamsRcd>();
}

/* virtual */ void ZDCQIE10Task::bookHistograms(DQMStore::IBooker& ib, edm::Run const& r, edm::EventSetup const& es) {
  DQTask::bookHistograms(ib, r, es);
  ib.cd();
  edm::ESHandle<HcalDbService> dbs = es.getHandle(hcalDbServiceToken_);
  _emap = dbs->getHcalMapping();
  // Book LED calibration channels from emap
  std::vector<HcalElectronicsId> eids = _emap->allElectronicsId();
  _ehashmap.initialize(_emap, electronicsmap::fD2EHashMap);

  //    BOOK HISTOGRAMS
  std::string histoname;
  histoname = "Crate28";
  ib.setCurrentFolder("Hcal/ZDCQIE10Task/DigiSize");
  _cDigiSize_Crate[0] = ib.book1DD(histoname.c_str(), histoname.c_str(), 21, -0.5, 20.5);
  _cDigiSize_Crate[0]->setAxisTitle("DigiSize", 1);
  _cDigiSize_Crate[0]->setAxisTitle("N", 2);

  histoname = "FED1136";
  ib.setCurrentFolder("Hcal/ZDCQIE10Task/DigiSize");
  _cDigiSize_FED[0] = ib.book1DD(histoname.c_str(), histoname.c_str(), 21, -0.5, 20.5);
  _cDigiSize_FED[0]->setAxisTitle("DigiSize", 1);
  _cDigiSize_FED[0]->setAxisTitle("N", 2);

  histoname = "ZDCM";
  ib.setCurrentFolder("Hcal/ZDCQIE10Task/ADC");
  _cADC_PM[0] = ib.book1DD(histoname.c_str(), histoname.c_str(), 256, 0, 256);
  _cADC_PM[0]->setAxisTitle("ADC", 1);
  _cADC_PM[0]->setAxisTitle("N", 2);

  histoname = "ZDCP";
  ib.setCurrentFolder("Hcal/ZDCQIE10Task/ADC");
  _cADC_PM[1] = ib.book1DD(histoname.c_str(), histoname.c_str(), 256, 0, 256);
  _cADC_PM[1]->setAxisTitle("ADC", 1);
  _cADC_PM[1]->setAxisTitle("N", 2);

  histoname = "ZDCM";
  ib.setCurrentFolder("Hcal/ZDCQIE10Task/ADCvsTS");
  _cADC_vs_TS_PM[0] = ib.book2DD(histoname.c_str(), histoname.c_str(), 6, 0, 6, 256, 0, 256);
  _cADC_vs_TS_PM[0]->setAxisTitle("TS", 1);
  _cADC_vs_TS_PM[0]->setAxisTitle("ADC", 2);

  histoname = "ZDCP";
  ib.setCurrentFolder("Hcal/ZDCQIE10Task/ADCvsTS");
  _cADC_vs_TS_PM[1] = ib.book2DD(histoname.c_str(), histoname.c_str(), 6, 0, 6, 256, 0, 256);
  _cADC_vs_TS_PM[1]->setAxisTitle("TS", 1);
  _cADC_vs_TS_PM[1]->setAxisTitle("ADC", 2);

  histoname = "Crate28";
  ib.setCurrentFolder("Hcal/ZDCQIE10Task/Occupancy/Crate");
  _cOccupancy_Crate[0] = ib.book2DD(histoname.c_str(), histoname.c_str(), 12, 0, 12, 144, 0, 144);
  _cOccupancy_Crate[0]->setAxisTitle("SlotuTCA", 1);
  _cOccupancy_Crate[0]->setAxisTitle("FiberuTCAFiberCh", 2);

  histoname = "Crate28S3";
  ib.setCurrentFolder("Hcal/ZDCQIE10Task/Occupancy/CrateSlote");
  _cOccupancy_CrateSlot[0] = ib.book2DD(histoname.c_str(), histoname.c_str(), 24, 0, 24, 6, 0, 6);
  _cOccupancy_CrateSlot[0]->setAxisTitle("SlotuTCA", 1);
  _cOccupancy_CrateSlot[0]->setAxisTitle("FiberCh", 2);

  histoname = "uTCA_for_FED1136";
  ib.setCurrentFolder("Hcal/ZDCQIE10Task/Occupancy/Electronics");
  _cOccupancy_ElectronicsuTCA[0] = ib.book1DD(histoname.c_str(), histoname.c_str(), 12, 0, 12);
  _cOccupancy_ElectronicsuTCA[0]->setAxisTitle("N", 1);
  _cOccupancy_ElectronicsuTCA[0]->setAxisTitle("SlotuTCA", 1);

  histoname = "ZDCP_Sum";
  ib.setCurrentFolder("Hcal/ZDCQIE10Task/Sums");
  _cZDC_SUMS[1] = ib.book1DD(histoname.c_str(), histoname.c_str(), 100, -1, 5000);
  _cZDC_SUMS[1]->setAxisTitle("GeV", 1);
  _cZDC_SUMS[1]->setAxisTitle("N", 2);

  histoname = "ZDCM_Sum";
  ib.setCurrentFolder("Hcal/ZDCQIE10Task/Sums");
  _cZDC_SUMS[0] = ib.book1DD(histoname.c_str(), histoname.c_str(), 100, -1, 5000);
  _cZDC_SUMS[0]->setAxisTitle("GeV", 1);
  _cZDC_SUMS[0]->setAxisTitle("N", 2);

  histoname = "ZDCM_SumBX";
  ib.setCurrentFolder("Hcal/ZDCQIE10Task/Sums");
  _cZDC_BXSUMS[0] = ib.book1DD(histoname.c_str(), histoname.c_str(), 3500, 0, 3500);
  _cZDC_BXSUMS[0]->setAxisTitle("globalBX", 1);
  _cZDC_BXSUMS[0]->setAxisTitle("GeV", 2);

  histoname = "ZDCP_SumBX";
  ib.setCurrentFolder("Hcal/ZDCQIE10Task/Sums");
  _cZDC_BXSUMS[1] = ib.book1DD(histoname.c_str(), histoname.c_str(), 3500, 0, 3500);
  _cZDC_BXSUMS[1]->setAxisTitle("globalBX", 1);
  _cZDC_BXSUMS[1]->setAxisTitle("GeV", 2);

  histoname = "ZDCM_EmuSumBX";
  ib.setCurrentFolder("Hcal/ZDCQIE10Task/Sums");
  _cZDC_BX_EmuSUMS[0] = ib.book1DD(histoname.c_str(), histoname.c_str(), 3500, 0, 3500);
  _cZDC_BX_EmuSUMS[0]->setAxisTitle("globalBX", 1);
  _cZDC_BX_EmuSUMS[0]->setAxisTitle("0-255 weighted output", 2);

  histoname = "ZDCP_EmuSumBX";
  ib.setCurrentFolder("Hcal/ZDCQIE10Task/Sums");
  _cZDC_BX_EmuSUMS[1] = ib.book1DD(histoname.c_str(), histoname.c_str(), 3500, 0, 3500);
  _cZDC_BX_EmuSUMS[1]->setAxisTitle("globalBX", 1);
  _cZDC_BX_EmuSUMS[1]->setAxisTitle("0-255 weighted output", 2);

  histoname = "CapIDs";
  ib.setCurrentFolder("Hcal/ZDCQIE10Task");
  _cZDC_CapIDS[0] = ib.book1DD(histoname.c_str(), histoname.c_str(), 4, 0, 4);
  _cZDC_CapIDS[0]->setAxisTitle("CapID", 1);
  _cZDC_CapIDS[0]->setAxisTitle("N", 2);

  histoname = "EM_M_Timings";
  ib.setCurrentFolder("Hcal/ZDCQIE10Task");
  _cZDC_EM_TM[0] = ib.book1DD(histoname.c_str(), histoname.c_str(), 10, 1, 11);
  _cZDC_EM_TM[0]->setAxisTitle("Channel", 1);
  _cZDC_EM_TM[0]->setAxisTitle("N", 2);
  _cZDC_EM_TM[0]->setBinLabel(1, "EM_M_1 good timing");
  _cZDC_EM_TM[0]->setBinLabel(2, "EM_M_1 bad timing");
  _cZDC_EM_TM[0]->setBinLabel(3, "EM_M_2 good timing");
  _cZDC_EM_TM[0]->setBinLabel(4, "EM_M_2 bad timing");
  _cZDC_EM_TM[0]->setBinLabel(5, "EM_M_3 good timing");
  _cZDC_EM_TM[0]->setBinLabel(6, "EM_M_3 bad timing");
  _cZDC_EM_TM[0]->setBinLabel(7, "EM_M_4 good timing");
  _cZDC_EM_TM[0]->setBinLabel(8, "EM_M_4 bad timing");
  _cZDC_EM_TM[0]->setBinLabel(9, "EM_M_5 good timing");
  _cZDC_EM_TM[0]->setBinLabel(10, "EM_M_5 bad timing");

  histoname = "EM_P_Timings";
  ib.setCurrentFolder("Hcal/ZDCQIE10Task");
  _cZDC_EM_TM[1] = ib.book1DD(histoname.c_str(), histoname.c_str(), 10, 1, 11);
  _cZDC_EM_TM[1]->setAxisTitle("Channel", 1);
  _cZDC_EM_TM[1]->setAxisTitle("N", 2);
  _cZDC_EM_TM[1]->setBinLabel(1, "EM_P_1 good timing");
  _cZDC_EM_TM[1]->setBinLabel(2, "EM_P_1 bad timing");
  _cZDC_EM_TM[1]->setBinLabel(3, "EM_P_2 good timing");
  _cZDC_EM_TM[1]->setBinLabel(4, "EM_P_2 bad timing");
  _cZDC_EM_TM[1]->setBinLabel(5, "EM_P_3 good timing");
  _cZDC_EM_TM[1]->setBinLabel(6, "EM_P_3 bad timing");
  _cZDC_EM_TM[1]->setBinLabel(7, "EM_P_4 good timing");
  _cZDC_EM_TM[1]->setBinLabel(8, "EM_P_4 bad timing");
  _cZDC_EM_TM[1]->setBinLabel(9, "EM_P_5 good timing");
  _cZDC_EM_TM[1]->setBinLabel(10, "EM_P_5 bad timing");

  histoname = "HAD_M_Timings";
  ib.setCurrentFolder("Hcal/ZDCQIE10Task");
  _cZDC_HAD_TM[0] = ib.book1DD(histoname.c_str(), histoname.c_str(), 8, 1, 9);
  _cZDC_HAD_TM[0]->setAxisTitle("Channel", 1);
  _cZDC_HAD_TM[0]->setAxisTitle("N", 2);
  _cZDC_HAD_TM[0]->setBinLabel(1, "HAD_M_1 good timing");
  _cZDC_HAD_TM[0]->setBinLabel(2, "HAD_M_1 bad timing");
  _cZDC_HAD_TM[0]->setBinLabel(3, "HAD_M_2 good timing");
  _cZDC_HAD_TM[0]->setBinLabel(4, "HAD_M_2 bad timing");
  _cZDC_HAD_TM[0]->setBinLabel(5, "HAD_M_3 good timing");
  _cZDC_HAD_TM[0]->setBinLabel(6, "HAD_M_3 bad timing");
  _cZDC_HAD_TM[0]->setBinLabel(7, "HAD_M_4 good timing");
  _cZDC_HAD_TM[0]->setBinLabel(8, "HAD_M_4 bad timing");

  histoname = "HAD_P_Timings";
  ib.setCurrentFolder("Hcal/ZDCQIE10Task");
  _cZDC_HAD_TM[1] = ib.book1DD(histoname.c_str(), histoname.c_str(), 8, 1, 9);
  _cZDC_HAD_TM[1]->setAxisTitle("Channel", 1);
  _cZDC_HAD_TM[1]->setAxisTitle("N", 2);
  _cZDC_HAD_TM[1]->setBinLabel(1, "HAD_P_1 good timing");
  _cZDC_HAD_TM[1]->setBinLabel(2, "HAD_P_1 bad timing");
  _cZDC_HAD_TM[1]->setBinLabel(3, "HAD_P_2 good timing");
  _cZDC_HAD_TM[1]->setBinLabel(4, "HAD_P_2 bad timing");
  _cZDC_HAD_TM[1]->setBinLabel(5, "HAD_P_3 good timing");
  _cZDC_HAD_TM[1]->setBinLabel(6, "HAD_P_3 bad timing");
  _cZDC_HAD_TM[1]->setBinLabel(7, "HAD_P_4 good timing");
  _cZDC_HAD_TM[1]->setBinLabel(8, "HAD_P_4 bad timing");

  //book histos per channel
  //std::string histoname;
  for (int channel = 1; channel < 6; channel++) {
    // EM Pos
    HcalZDCDetId didp(HcalZDCDetId::EM, true, channel);

    histoname = "EM_P_" + std::to_string(channel);
    ib.setCurrentFolder("Hcal/ZDCQIE10Task/ADC_perChannel");
    _cADC_EChannel[didp()] = ib.book1DD(histoname.c_str(), histoname.c_str(), 256, 0, 256);
    _cADC_EChannel[didp()]->setAxisTitle("ADC", 1);
    _cADC_EChannel[didp()]->setAxisTitle("N", 2);
    ib.setCurrentFolder("Hcal/ZDCQIE10Task/ADC_vs_TS_perChannel");
    _cADC_vs_TS_EChannel[didp()] = ib.book1DD(histoname.c_str(), histoname.c_str(), 6, 0, 6);
    _cADC_vs_TS_EChannel[didp()]->setAxisTitle("TS", 1);
    _cADC_vs_TS_EChannel[didp()]->setAxisTitle("sum ADC", 2);

    histoname = "EM_P_" + std::to_string(channel);
    ib.setCurrentFolder("Hcal/ZDCQIE10Task/fC_perChannel");
    _cfC_EChannel[didp()] = ib.book1DD(histoname.c_str(), histoname.c_str(), 100, 0, 8000);
    _cfC_EChannel[didp()]->setAxisTitle("fC", 1);
    _cfC_EChannel[didp()]->setAxisTitle("N", 2);
    ib.setCurrentFolder("Hcal/ZDCQIE10Task/fC_vs_TS_perChannel");
    _cfC_vs_TS_EChannel[didp()] = ib.book1DD(histoname.c_str(), histoname.c_str(), 6, 0, 6);
    _cfC_vs_TS_EChannel[didp()]->setAxisTitle("TS", 1);
    _cfC_vs_TS_EChannel[didp()]->setAxisTitle("sum fC", 2);

    histoname = "EM_P_" + std::to_string(channel);
    _cTDC_EChannel[didp()] = ib.book1DD(histoname.c_str(), histoname.c_str(), 50, 1, 50);
    _cTDC_EChannel[didp()]->setAxisTitle("TDC", 1);
    _cTDC_EChannel[didp()]->setAxisTitle("N", 2);

    // EM Minus
    HcalZDCDetId didm(HcalZDCDetId::EM, false, channel);

    histoname = "EM_M_" + std::to_string(channel);
    ib.setCurrentFolder("Hcal/ZDCQIE10Task/ADC_perChannel");
    _cADC_EChannel[didm()] = ib.book1DD(histoname.c_str(), histoname.c_str(), 256, 0, 256);
    _cADC_EChannel[didm()]->setAxisTitle("ADC", 1);
    _cADC_EChannel[didm()]->setAxisTitle("N", 2);
    ib.setCurrentFolder("Hcal/ZDCQIE10Task/ADC_vs_TS_perChannel");
    _cADC_vs_TS_EChannel[didm()] = ib.book1DD(histoname.c_str(), histoname.c_str(), 6, 0, 6);
    _cADC_vs_TS_EChannel[didm()]->setAxisTitle("TS", 1);
    _cADC_vs_TS_EChannel[didm()]->setAxisTitle("sum ADC", 2);

    histoname = "EM_M_" + std::to_string(channel);
    ib.setCurrentFolder("Hcal/ZDCQIE10Task/fC_perChannel");
    _cfC_EChannel[didm()] = ib.book1DD(histoname.c_str(), histoname.c_str(), 100, 0, 8000);
    _cfC_EChannel[didm()]->setAxisTitle("fC", 1);
    _cfC_EChannel[didm()]->setAxisTitle("N", 2);
    ib.setCurrentFolder("Hcal/ZDCQIE10Task/fC_vs_TS_perChannel");
    _cfC_vs_TS_EChannel[didm()] = ib.book1DD(histoname.c_str(), histoname.c_str(), 6, 0, 6);
    _cfC_vs_TS_EChannel[didm()]->setAxisTitle("TS", 1);
    _cfC_vs_TS_EChannel[didm()]->setAxisTitle("sum fC", 2);

    histoname = "EM_M_" + std::to_string(channel);
    _cTDC_EChannel[didm()] = ib.book1DD(histoname.c_str(), histoname.c_str(), 50, 1, 50);
    _cTDC_EChannel[didm()]->setAxisTitle("TDC", 1);
    _cTDC_EChannel[didm()]->setAxisTitle("N", 2);
  }

  for (int channel = 1; channel < 5; channel++) {
    // HAD Pos
    HcalZDCDetId didp(HcalZDCDetId::HAD, true, channel);

    histoname = "HAD_P_" + std::to_string(channel);
    ib.setCurrentFolder("Hcal/ZDCQIE10Task/ADC_perChannel");
    _cADC_EChannel[didp()] = ib.book1DD(histoname.c_str(), histoname.c_str(), 256, 0, 256);
    _cADC_EChannel[didp()]->setAxisTitle("ADC", 1);
    _cADC_EChannel[didp()]->setAxisTitle("N", 2);
    ib.setCurrentFolder("Hcal/ZDCQIE10Task/ADC_vs_TS_perChannel");
    _cADC_vs_TS_EChannel[didp()] = ib.book1DD(histoname.c_str(), histoname.c_str(), 6, 0, 6);
    _cADC_vs_TS_EChannel[didp()]->setAxisTitle("TS", 1);
    _cADC_vs_TS_EChannel[didp()]->setAxisTitle("sum ADC", 2);

    histoname = "HAD_P_" + std::to_string(channel);
    ib.setCurrentFolder("Hcal/ZDCQIE10Task/fC_perChannel");
    _cfC_EChannel[didp()] = ib.book1DD(histoname.c_str(), histoname.c_str(), 100, 0, 8000);
    _cfC_EChannel[didp()]->setAxisTitle("fC", 1);
    _cfC_EChannel[didp()]->setAxisTitle("N", 2);
    ib.setCurrentFolder("Hcal/ZDCQIE10Task/fC_vs_TS_perChannel");
    _cfC_vs_TS_EChannel[didp()] = ib.book1DD(histoname.c_str(), histoname.c_str(), 6, 0, 6);
    _cfC_vs_TS_EChannel[didp()]->setAxisTitle("TS", 1);
    _cfC_vs_TS_EChannel[didp()]->setAxisTitle("sum fC", 2);

    histoname = "HAD_P_" + std::to_string(channel);
    _cTDC_EChannel[didp()] = ib.book1DD(histoname.c_str(), histoname.c_str(), 50, 1, 50);
    _cTDC_EChannel[didp()]->setAxisTitle("TDC", 1);
    _cTDC_EChannel[didp()]->setAxisTitle("N", 2);

    // HAD Minus
    HcalZDCDetId didm(HcalZDCDetId::HAD, false, channel);

    histoname = "HAD_M_" + std::to_string(channel);
    ib.setCurrentFolder("Hcal/ZDCQIE10Task/ADC_perChannel");
    _cADC_EChannel[didm()] = ib.book1DD(histoname.c_str(), histoname.c_str(), 256, 0, 256);
    _cADC_EChannel[didm()]->setAxisTitle("ADC", 1);
    _cADC_EChannel[didm()]->setAxisTitle("N", 2);
    ib.setCurrentFolder("Hcal/ZDCQIE10Task/ADC_vs_TS_perChannel");
    _cADC_vs_TS_EChannel[didm()] = ib.book1DD(histoname.c_str(), histoname.c_str(), 6, 0, 6);
    _cADC_vs_TS_EChannel[didm()]->setAxisTitle("TS", 1);
    _cADC_vs_TS_EChannel[didm()]->setAxisTitle("sum ADC", 2);

    histoname = "HAD_M_" + std::to_string(channel);
    ib.setCurrentFolder("Hcal/ZDCQIE10Task/fC_perChannel");
    _cfC_EChannel[didm()] = ib.book1DD(histoname.c_str(), histoname.c_str(), 100, 0, 8000);
    _cfC_EChannel[didm()]->setAxisTitle("fC", 1);
    _cfC_EChannel[didm()]->setAxisTitle("N", 2);
    ib.setCurrentFolder("Hcal/ZDCQIE10Task/fC_vs_TS_perChannel");
    _cfC_vs_TS_EChannel[didm()] = ib.book1DD(histoname.c_str(), histoname.c_str(), 6, 0, 6);
    _cfC_vs_TS_EChannel[didm()]->setAxisTitle("TS", 1);
    _cfC_vs_TS_EChannel[didm()]->setAxisTitle("sum fC", 2);

    histoname = "HAD_M_" + std::to_string(channel);
    _cTDC_EChannel[didm()] = ib.book1DD(histoname.c_str(), histoname.c_str(), 50, 1, 50);
    _cTDC_EChannel[didm()]->setAxisTitle("TDC", 1);
    _cTDC_EChannel[didm()]->setAxisTitle("N", 2);
  }

  for (int channel = 1; channel < 17; channel++) {
    // RPD Pos
    HcalZDCDetId didp(HcalZDCDetId::RPD, true, channel);

    histoname = "RPD_P_" + std::to_string(channel);
    ib.setCurrentFolder("Hcal/ZDCQIE10Task/ADC_perChannel");
    _cADC_EChannel[didp()] = ib.book1DD(histoname.c_str(), histoname.c_str(), 256, 0, 256);
    _cADC_EChannel[didp()]->setAxisTitle("ADC", 1);
    _cADC_EChannel[didp()]->setAxisTitle("N", 2);
    ib.setCurrentFolder("Hcal/ZDCQIE10Task/ADC_vs_TS_perChannel");
    _cADC_vs_TS_EChannel[didp()] = ib.book1DD(histoname.c_str(), histoname.c_str(), 6, 0, 6);
    _cADC_vs_TS_EChannel[didp()]->setAxisTitle("TS", 1);
    _cADC_vs_TS_EChannel[didp()]->setAxisTitle("sum ADC", 2);

    histoname = "RPD_P_" + std::to_string(channel);
    ib.setCurrentFolder("Hcal/ZDCQIE10Task/fC_perChannel");
    _cfC_EChannel[didp()] = ib.book1DD(histoname.c_str(), histoname.c_str(), 100, 0, 8000);
    _cfC_EChannel[didp()]->setAxisTitle("fC", 1);
    _cfC_EChannel[didp()]->setAxisTitle("N", 2);
    ib.setCurrentFolder("Hcal/ZDCQIE10Task/fC_vs_TS_perChannel");
    _cfC_vs_TS_EChannel[didp()] = ib.book1DD(histoname.c_str(), histoname.c_str(), 6, 0, 6);
    _cfC_vs_TS_EChannel[didp()]->setAxisTitle("TS", 1);
    _cfC_vs_TS_EChannel[didp()]->setAxisTitle("sum fC", 2);

    histoname = "RPD_P_" + std::to_string(channel);
    _cTDC_EChannel[didp()] = ib.book1DD(histoname.c_str(), histoname.c_str(), 50, 1, 50);
    _cTDC_EChannel[didp()]->setAxisTitle("TDC", 1);
    _cTDC_EChannel[didp()]->setAxisTitle("N", 2);

    // RPD Minus
    HcalZDCDetId didm(HcalZDCDetId::RPD, false, channel);
    histoname = "RPD_M_" + std::to_string(channel);

    ib.setCurrentFolder("Hcal/ZDCQIE10Task/ADC_perChannel");
    _cADC_EChannel[didm()] = ib.book1DD(histoname.c_str(), histoname.c_str(), 256, 0, 256);
    _cADC_EChannel[didm()]->setAxisTitle("ADC", 1);
    _cADC_EChannel[didm()]->setAxisTitle("N", 2);
    ib.setCurrentFolder("Hcal/ZDCQIE10Task/ADC_vs_TS_perChannel");
    _cADC_vs_TS_EChannel[didm()] = ib.book1DD(histoname.c_str(), histoname.c_str(), 6, 0, 6);
    _cADC_vs_TS_EChannel[didm()]->setAxisTitle("TS", 1);
    _cADC_vs_TS_EChannel[didm()]->setAxisTitle("sum ADC", 2);

    ib.setCurrentFolder("Hcal/ZDCQIE10Task/fC_perChannel");
    _cfC_EChannel[didm()] = ib.book1DD(histoname.c_str(), histoname.c_str(), 100, 0, 8000);
    _cfC_EChannel[didm()]->setAxisTitle("fC", 1);
    _cfC_EChannel[didm()]->setAxisTitle("N", 2);
    ib.setCurrentFolder("Hcal/ZDCQIE10Task/fC_vs_TS_perChannel");
    _cfC_vs_TS_EChannel[didm()] = ib.book1DD(histoname.c_str(), histoname.c_str(), 6, 0, 6);
    _cfC_vs_TS_EChannel[didm()]->setAxisTitle("TS", 1);
    _cfC_vs_TS_EChannel[didm()]->setAxisTitle("sum fC", 2);

    histoname = "RPD_M_" + std::to_string(channel);
    _cTDC_EChannel[didm()] = ib.book1DD(histoname.c_str(), histoname.c_str(), 50, 1, 50);
    _cTDC_EChannel[didm()]->setAxisTitle("TDC", 1);
    _cTDC_EChannel[didm()]->setAxisTitle("N", 2);
  }
}

/* virtual */
void ZDCQIE10Task::_process(edm::Event const& e, edm::EventSetup const& es) {
  const HcalTopology& htopo = es.getData(htopoToken_);
  const HcalLongRecoParams& p = es.getData(paramsToken_);
  longRecoParams_ = std::make_unique<HcalLongRecoParams>(p);
  longRecoParams_->setTopo(&htopo);

  int bx = e.bunchCrossing();

  edm::Handle<BXVector<l1t::EtSum> > sums;
  e.getByToken(sumToken_, sums);

  int startBX = sums->getFirstBX();

  for (int ibx = startBX; ibx <= sums->getLastBX(); ++ibx) {
    for (auto itr = sums->begin(ibx); itr != sums->end(ibx); ++itr) {
      l1t::EtSum::EtSumType type = itr->getType();

      if (type == l1t::EtSum::EtSumType::kZDCP) {
        if (ibx == 0)
          _cZDC_BX_EmuSUMS[1]->Fill(bx, itr->hwPt());
      }
      if (type == l1t::EtSum::EtSumType::kZDCM) {
        if (ibx == 0)
          _cZDC_BX_EmuSUMS[0]->Fill(bx, itr->hwPt());
      }
    }
  }

  edm::Handle<QIE10DigiCollection> digis;
  if (!e.getByToken(_tokQIE10, digis))
    edm::LogError("Collection QIE10DigiCollection for ZDC isn't available" + _tagQIE10.label() + " " +
                  _tagQIE10.instance());

  double HADM_sum = 0, HADP_sum = 0, EMM_sum = 0, EMP_sum = 0;
  double HADM_tot_sum = 0, HADP_tot_sum = 0, EMM_tot_sum = 0, EMP_tot_sum = 0;

  for (auto it = digis->begin(); it != digis->end(); it++) {
    const QIE10DataFrame digi = static_cast<const QIE10DataFrame>(*it);

    HcalZDCDetId const& did = digi.detid();

    uint32_t rawid = _ehashmap.lookup(did);
    if (rawid == 0) {
      continue;
    }
    HcalElectronicsId const& eid(rawid);

    _cDigiSize_Crate[0]->Fill(digi.samples());

    if (_ptype != fOffline) {  // hidefed2crate

      _cDigiSize_FED[0]->Fill(digi.samples());

      _cOccupancy_ElectronicsuTCA[0]->Fill(eid.slot());
    }
    if (_ptype == fOnline || _ptype == fLocal) {
      _cOccupancy_Crate[0]->Fill(eid.slot(), eid.fiberIndex());
      _cOccupancy_CrateSlot[0]->Fill(eid.slot(), eid.fiberChanId());
    }

    double sample_ZDCm_TS1 = 0, sample_ZDCp_TS1 = 0, sample_ZDCm_TS2 = 0, sample_ZDCp_TS2 = 0;
    double sample[2][6] = {{0}};

    for (int i = 0; i < digi.samples(); i++) {
      // ZDC Plus
      if (did.zside() > 0) {
        _cADC_PM[1]->Fill(digi[i].adc());
        _cADC_vs_TS_PM[1]->Fill(i, digi[i].adc());
        sample[1][i] = constants::adc2fC[digi[i].adc()];
        if (i == 1) {
          sample_ZDCp_TS1 = constants::adc2fC[digi[i].adc()];
        }
        if (i == 2) {
          sample_ZDCp_TS2 = constants::adc2fC[digi[i].adc()];
        }

      }
      // ZDC Minus
      else {
        _cADC_PM[0]->Fill(digi[i].adc());
        _cADC_vs_TS_PM[0]->Fill(i, digi[i].adc());
        sample[0][i] = constants::adc2fC[digi[i].adc()];
        if (i == 1) {
          sample_ZDCm_TS1 = constants::adc2fC[digi[i].adc()];
        }
        if (i == 2) {
          sample_ZDCm_TS2 = constants::adc2fC[digi[i].adc()];
        }
      }

      // iter over all samples
      if (_cADC_EChannel.find(did()) != _cADC_EChannel.end()) {
        _cADC_EChannel[did()]->Fill(digi[i].adc());
        _cfC_EChannel[did()]->Fill(constants::adc2fC[digi[i].adc()]);
        _cTDC_EChannel[did()]->Fill(digi[i].le_tdc());
      }
      if (_cADC_vs_TS_EChannel.find(did()) != _cADC_vs_TS_EChannel.end()) {
        _cADC_vs_TS_EChannel[did()]->Fill(i, digi[i].adc());
        _cfC_vs_TS_EChannel[did()]->Fill(i, constants::adc2fC[digi[i].adc()]);
      }
      _cZDC_CapIDS[0]->Fill(digi[i].capid());
    }

    if (did.section() == 1) {
      if ((did.channel() > 0) && (did.channel() < 6)) {
        EMM_sum += (sample_ZDCm_TS2 - sample_ZDCm_TS1);
        EMP_sum += (sample_ZDCp_TS2 - sample_ZDCp_TS1);
        for (int k = 0; k < 6; k++) {
          EMP_tot_sum += sample[1][k];
          EMM_tot_sum += sample[0][k];
        }
        if (did.zside() == -1) {
          if (sample[0][2] > (0.0 * EMM_tot_sum)) {
            _cZDC_EM_TM[0]->Fill((did.channel() * 2) - 1);
          } else {
            _cZDC_EM_TM[0]->Fill((did.channel() * 2));
          }
        }
        if (did.zside() == 1) {
          if (sample[1][2] > (0.0 * EMP_tot_sum)) {
            _cZDC_EM_TM[1]->Fill((did.channel() * 2) - 1);
          } else {
            _cZDC_EM_TM[1]->Fill((did.channel() * 2));
          }
        }
      }
    }
    if (did.section() == 2) {
      if ((did.channel() > 0) && (did.channel() < 5)) {
        HADP_sum += (sample_ZDCp_TS2 - sample_ZDCp_TS1);
        HADM_sum += (sample_ZDCm_TS2 - sample_ZDCm_TS1);
        for (int k = 0; k < 6; k++) {
          HADP_tot_sum += sample[1][k];
          HADM_tot_sum += sample[0][k];
        }
        if (did.zside() == -1) {
          if (sample[0][2] > (0.0 * HADM_tot_sum)) {
            _cZDC_HAD_TM[0]->Fill((did.channel() * 2) - 1);
          } else {
            _cZDC_HAD_TM[0]->Fill((did.channel() * 2));
          }
        }
        if (did.zside() == 1) {
          if (sample[1][2] > (0.0 * HADP_tot_sum)) {
            _cZDC_HAD_TM[1]->Fill((did.channel() * 2) - 1);
          } else {
            _cZDC_HAD_TM[1]->Fill((did.channel() * 2));
          }
        }
      }
    }
  }

  // Hardcoded callibrations currently
  _cZDC_SUMS[0]->Fill(((EMM_sum * 0.1) + HADM_sum) * 0.5031);
  _cZDC_SUMS[1]->Fill(((EMP_sum * 0.12) + HADP_sum) * 0.9397);
  _cZDC_BXSUMS[0]->Fill(bx, ((EMM_sum * 0.1) + HADM_sum) * 0.5031);
  _cZDC_BXSUMS[1]->Fill(bx, ((EMP_sum * 0.12) + HADP_sum) * 0.9397);
}

DEFINE_FWK_MODULE(ZDCQIE10Task);
