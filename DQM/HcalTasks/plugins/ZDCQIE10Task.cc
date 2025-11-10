#include "DQM/HcalTasks/interface/ZDCQIE10Task.h"
using namespace std;
using namespace hcaldqm;
using namespace hcaldqm::constants;
using namespace hcaldqm::filter;

ZDCQIE10Task::ZDCQIE10Task(edm::ParameterSet const& ps)

    : DQTask(ps), hcalDbServiceToken_(esConsumes<HcalDbService, HcalDbRecord, edm::Transition::BeginRun>()) {
  //tags data for digis and TPs
  _tagQIE10 = ps.getUntrackedParameter<edm::InputTag>("tagQIE10", edm::InputTag("hcalDigis", "ZDC"));
  _tokQIE10 = consumes<QIE10DigiCollection>(_tagQIE10);
  _tagData = ps.getUntrackedParameter<edm::InputTag>("tagData", edm::InputTag("hcalDigis"));
  sumTokenData_ = consumes<HcalTrigPrimDigiCollection>(_tagData);

  // emulated sums
  sumTag = ps.getUntrackedParameter<edm::InputTag>("etSumTag", edm::InputTag("etSumZdcProducer", ""));
  sumToken_ = consumes<l1t::EtSumBxCollection>(sumTag);

  // unpacked sums
  sumTagUnpacked =
      ps.getUntrackedParameter<edm::InputTag>("etSumTagUnpacked", edm::InputTag("gtStage2Digis", "EtSumZDC"));
  sumTokenUnpacked_ = consumes<l1t::EtSumBxCollection>(sumTagUnpacked);

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

  // create variable binning for TP sum histograms
  std::vector<double> varbins;
  // -1 - 64 : 65 1-unit bins
  // 64 - 128 : 32 2-unit bins
  // 128 - 256 : 32 4-unit bins
  // 256 - 512 : 32 8-unit bins
  // 512 - 1008 : 31 16-unit bins
  // 1008 - 1023: 1 bin
  // 1023 - 1024: 1 bin
  for (int i = -1; i < 64; i += 1)
    varbins.push_back(i);
  for (int i = 64; i < 128; i += 2)
    varbins.push_back(i);
  for (int i = 128; i < 256; i += 4)
    varbins.push_back(i);
  for (int i = 256; i < 512; i += 8)
    varbins.push_back(i);
  for (int i = 512; i <= 1008; i += 16)
    varbins.push_back(i);

  // add additional bins
  varbins.push_back(1023);
  varbins.push_back(1024);

  // Emu Sum TP vs. Data Sum TP
  histoname = "ZDCM_EmuSumTP_DataSumTP";
  ib.setCurrentFolder("Hcal/ZDCQIE10Task/TPs");

  TH2D* varBinningTH2D = new TH2D(
      histoname.c_str(), histoname.c_str(), varbins.size() - 1, varbins.data(), varbins.size() - 1, varbins.data());

  _cZDC_EmuSumTP_DataSum[0] = ib.book2DD(histoname.c_str(), varBinningTH2D);
  _cZDC_EmuSumTP_DataSum[0]->setAxisTitle("Emulated TP Sum (Online Counts)", 2);

  histoname = "ZDCP_EmuSumTP_DataSum";
  ib.setCurrentFolder("Hcal/ZDCQIE10Task/TPs");
  _cZDC_EmuSumTP_DataSum[1] = ib.book2DD(histoname.c_str(), varBinningTH2D);
  _cZDC_EmuSumTP_DataSum[1]->setAxisTitle("Data TP Sum (Online Counts)", 1);
  _cZDC_EmuSumTP_DataSum[1]->setAxisTitle("Emulated TP Sum (Online Counts)", 2);

  // Emu Sum TP vs. L1rcvd Sum TP
  histoname = "ZDCM_EmuSumTP_L1rcvdSumTP";
  ib.setCurrentFolder("Hcal/ZDCQIE10Task/TPs");

  TH2D* varBinningTH2D_EmuL1rcvd = new TH2D(
      histoname.c_str(), histoname.c_str(), varbins.size() - 1, varbins.data(), varbins.size() - 1, varbins.data());

  _cZDC_EmuSumTP_L1rcvdSum[0] = ib.book2DD(histoname.c_str(), varBinningTH2D_EmuL1rcvd);
  _cZDC_EmuSumTP_L1rcvdSum[0]->setAxisTitle("Emulated TP Sum (Online Counts)", 2);

  histoname = "ZDCP_EmuSumTP_L1rcvdSumTP";
  ib.setCurrentFolder("Hcal/ZDCQIE10Task/TPs");
  _cZDC_EmuSumTP_L1rcvdSum[1] = ib.book2DD(histoname.c_str(), varBinningTH2D_EmuL1rcvd);
  _cZDC_EmuSumTP_L1rcvdSum[1]->setAxisTitle("L1 rcvd TP Sum (Online Counts)", 1);
  _cZDC_EmuSumTP_L1rcvdSum[1]->setAxisTitle("Emulated TP Sum (Online Counts)", 2);

  // L1rcvd Sum TP vs. Data Sum TP
  histoname = "ZDCM_L1rcvdSumTP_DataSumTP";
  ib.setCurrentFolder("Hcal/ZDCQIE10Task/TPs");

  TH2D* varBinningTH2D_L1rcvdData = new TH2D(
      histoname.c_str(), histoname.c_str(), varbins.size() - 1, varbins.data(), varbins.size() - 1, varbins.data());

  _cZDC_L1rcvdSumTP_DataSum[0] = ib.book2DD(histoname.c_str(), varBinningTH2D_L1rcvdData);
  _cZDC_L1rcvdSumTP_DataSum[0]->setAxisTitle("L1 rcvd TP Sum (Online Counts)", 2);

  histoname = "ZDCP_L1rcvdSumTP_DataSumTP";
  ib.setCurrentFolder("Hcal/ZDCQIE10Task/TPs");
  _cZDC_L1rcvdSumTP_DataSum[1] = ib.book2DD(histoname.c_str(), varBinningTH2D_L1rcvdData);
  _cZDC_L1rcvdSumTP_DataSum[1]->setAxisTitle("Data TP Sum (Online Counts)", 1);
  _cZDC_L1rcvdSumTP_DataSum[1]->setAxisTitle("L1 rcvd TP Sum (Online Counts)", 2);

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
    ib.setCurrentFolder("Hcal/ZDCQIE10Task/TDC_perChannel");
    _cTDC_EChannel[didp()] = ib.book1DD(histoname.c_str(), histoname.c_str(), 150, 0, 150);
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
    ib.setCurrentFolder("Hcal/ZDCQIE10Task/TDC_perChannel");
    _cTDC_EChannel[didm()] = ib.book1DD(histoname.c_str(), histoname.c_str(), 150, 0, 150);
    _cTDC_EChannel[didm()]->setAxisTitle("TDC", 1);
    _cTDC_EChannel[didm()]->setAxisTitle("N", 2);
  }

  // adding 6 FSC channels on z- side (labeled as EM7_minus - EM12_minus)
  for (int channel = 7; channel < 13; channel++) {
    // FSC Minus
    HcalZDCDetId didm(HcalZDCDetId::EM, false, channel);

    std::vector<std::string> stationString = {
        "2_M_Top", "2_M_Bottom", "3_M_BottomLeft", "3_M_BottomRight", "3_M_TopLeft", "3_M_TopRight"};

    histoname = "FSC" + stationString.at(channel - 7);
    ib.setCurrentFolder("Hcal/ZDCQIE10Task/ADC_perChannel");
    _cADC_EChannel[didm()] = ib.book1DD(histoname.c_str(), histoname.c_str(), 256, 0, 256);
    _cADC_EChannel[didm()]->setAxisTitle("ADC", 1);
    _cADC_EChannel[didm()]->setAxisTitle("N", 2);
    ib.setCurrentFolder("Hcal/ZDCQIE10Task/ADC_vs_TS_perChannel");
    _cADC_vs_TS_EChannel[didm()] = ib.book1DD(histoname.c_str(), histoname.c_str(), 6, 0, 6);
    _cADC_vs_TS_EChannel[didm()]->setAxisTitle("TS", 1);
    _cADC_vs_TS_EChannel[didm()]->setAxisTitle("sum ADC", 2);

    histoname = "FSC" + stationString.at(channel - 7);
    ib.setCurrentFolder("Hcal/ZDCQIE10Task/fC_perChannel");
    _cfC_EChannel[didm()] = ib.book1DD(histoname.c_str(), histoname.c_str(), 100, 0, 8000);
    _cfC_EChannel[didm()]->setAxisTitle("fC", 1);
    _cfC_EChannel[didm()]->setAxisTitle("N", 2);
    ib.setCurrentFolder("Hcal/ZDCQIE10Task/fC_vs_TS_perChannel");
    _cfC_vs_TS_EChannel[didm()] = ib.book1DD(histoname.c_str(), histoname.c_str(), 6, 0, 6);
    _cfC_vs_TS_EChannel[didm()]->setAxisTitle("TS", 1);
    _cfC_vs_TS_EChannel[didm()]->setAxisTitle("sum fC", 2);

    histoname = "FSC" + stationString.at(channel - 7);
    ib.setCurrentFolder("Hcal/ZDCQIE10Task/TDC_perChannel");
    _cTDC_EChannel[didm()] = ib.book1DD(histoname.c_str(), histoname.c_str(), 150, 0, 150);
    _cTDC_EChannel[didm()]->setAxisTitle("TDC", 1);
    _cTDC_EChannel[didm()]->setAxisTitle("N", 2);

    // FSC plus
    HcalZDCDetId didp(HcalZDCDetId::EM, true, channel);

    std::vector<std::string> stationStringP = {
        "2_P_Top", "2_P_Bottom", "3_P_BottomLeft", "3_P_BottomRight", "3_P_TopLeft", "3_P_TopRight"};

    histoname = "FSC" + stationStringP.at(channel - 7);
    ib.setCurrentFolder("Hcal/ZDCQIE10Task/ADC_perChannel");
    _cADC_EChannel[didp()] = ib.book1DD(histoname.c_str(), histoname.c_str(), 256, 0, 256);
    _cADC_EChannel[didp()]->setAxisTitle("ADC", 1);
    _cADC_EChannel[didp()]->setAxisTitle("N", 2);
    ib.setCurrentFolder("Hcal/ZDCQIE10Task/ADC_vs_TS_perChannel");
    _cADC_vs_TS_EChannel[didp()] = ib.book1DD(histoname.c_str(), histoname.c_str(), 6, 0, 6);
    _cADC_vs_TS_EChannel[didp()]->setAxisTitle("TS", 1);
    _cADC_vs_TS_EChannel[didp()]->setAxisTitle("sum ADC", 2);

    histoname = "FSC" + stationStringP.at(channel - 7);
    ib.setCurrentFolder("Hcal/ZDCQIE10Task/fC_perChannel");
    _cfC_EChannel[didp()] = ib.book1DD(histoname.c_str(), histoname.c_str(), 100, 0, 8000);
    _cfC_EChannel[didp()]->setAxisTitle("fC", 1);
    _cfC_EChannel[didp()]->setAxisTitle("N", 2);
    ib.setCurrentFolder("Hcal/ZDCQIE10Task/fC_vs_TS_perChannel");
    _cfC_vs_TS_EChannel[didp()] = ib.book1DD(histoname.c_str(), histoname.c_str(), 6, 0, 6);
    _cfC_vs_TS_EChannel[didp()]->setAxisTitle("TS", 1);
    _cfC_vs_TS_EChannel[didp()]->setAxisTitle("sum fC", 2);

    histoname = "FSC" + stationStringP.at(channel - 7);
    ib.setCurrentFolder("Hcal/ZDCQIE10Task/TDC_perChannel");
    _cTDC_EChannel[didp()] = ib.book1DD(histoname.c_str(), histoname.c_str(), 150, 0, 150);
    _cTDC_EChannel[didp()]->setAxisTitle("TDC", 1);
    _cTDC_EChannel[didp()]->setAxisTitle("N", 2);
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
    ib.setCurrentFolder("Hcal/ZDCQIE10Task/TDC_perChannel");
    _cTDC_EChannel[didp()] = ib.book1DD(histoname.c_str(), histoname.c_str(), 150, 0, 150);
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
    _cTDC_EChannel[didm()] = ib.book1DD(histoname.c_str(), histoname.c_str(), 150, 0, 150);
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
    ib.setCurrentFolder("Hcal/ZDCQIE10Task/TDC_perChannel");
    _cTDC_EChannel[didp()] = ib.book1DD(histoname.c_str(), histoname.c_str(), 150, 1, 150);
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
    ib.setCurrentFolder("Hcal/ZDCQIE10Task/TDC_perChannel");
    _cTDC_EChannel[didm()] = ib.book1DD(histoname.c_str(), histoname.c_str(), 150, 1, 150);
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

  // to-do: if the TP is missing, this fills with -1
  double emulatedSumP = -1.0;
  double emulatedSumM = -1.0;
  for (int ibx = startBX; ibx <= sums->getLastBX(); ++ibx) {
    for (auto itr = sums->begin(ibx); itr != sums->end(ibx); ++itr) {
      l1t::EtSum::EtSumType type = itr->getType();

      if (type == l1t::EtSum::EtSumType::kZDCP) {
        if (ibx == 0) {
          _cZDC_BX_EmuSUMS[1]->Fill(bx, itr->hwPt());
          emulatedSumP = itr->hwPt();
        }
      }
      if (type == l1t::EtSum::EtSumType::kZDCM) {
        if (ibx == 0) {
          _cZDC_BX_EmuSUMS[0]->Fill(bx, itr->hwPt());
          emulatedSumM = itr->hwPt();
        }
      }
    }
  }

  edm::Handle<BXVector<l1t::EtSum> > unpacked_sums;
  e.getByToken(sumTokenUnpacked_, unpacked_sums);
  int startBX_Unpacked = unpacked_sums->getFirstBX();
  double unpackedSumP = -1.0;
  double unpackedSumM = -1.0;
  for (int ibx = startBX_Unpacked; ibx <= unpacked_sums->getLastBX(); ++ibx) {
    for (auto itr = unpacked_sums->begin(ibx); itr != unpacked_sums->end(ibx); ++itr) {
      l1t::EtSum::EtSumType type = itr->getType();

      if (type == l1t::EtSum::EtSumType::kZDCP) {
        if (ibx == 0) {
          unpackedSumP = itr->hwPt();
        }
      }
      if (type == l1t::EtSum::EtSumType::kZDCM) {
        if (ibx == 0) {
          unpackedSumM = itr->hwPt();
        }
      }
    }
  }

  edm::Handle<HcalTrigPrimDigiCollection> data_sums;
  e.getByToken(sumTokenData_, data_sums);
  double dataSumP = -1.0;
  double dataSumM = -1.0;
  for (HcalTrigPrimDigiCollection::const_iterator it = data_sums->begin(); it != data_sums->end(); ++it) {
    //	Explicit check on the DetIds present in the Collection
    HcalTrigTowerDetId tid = it->id();

    // ZDCp TP
    if (tid.ieta() >= kZDCAbsIEta && tid.iphi() == kZDCiEtSumsIPhi) {
      // need to convert to the actual dynamic range of the ZDC sums
      dataSumP = it->t0().raw() & kZDCiEtSumMaxValue;
    }
    // ZDCm TP
    if (tid.ieta() <= -kZDCAbsIEta && tid.iphi() == kZDCiEtSumsIPhi) {
      // need to convert to the actual dynamic range of the ZDC sums
      dataSumM = it->t0().raw() & kZDCiEtSumMaxValue;
    }
  }

  // now fill the unpacked and emulator comparison histogram
  _cZDC_EmuSumTP_DataSum[0]->Fill(dataSumM, emulatedSumM);
  _cZDC_EmuSumTP_DataSum[1]->Fill(dataSumP, emulatedSumP);
  _cZDC_EmuSumTP_L1rcvdSum[0]->Fill(unpackedSumM, emulatedSumM);
  _cZDC_EmuSumTP_L1rcvdSum[1]->Fill(unpackedSumP, emulatedSumP);
  _cZDC_L1rcvdSumTP_DataSum[0]->Fill(dataSumM, unpackedSumM);
  _cZDC_L1rcvdSumTP_DataSum[1]->Fill(dataSumP, unpackedSumP);

  edm::Handle<QIE10DigiCollection> digis;
  if (!e.getByToken(_tokQIE10, digis))
    edm::LogError("Collection QIE10DigiCollection for ZDC isn't available" + _tagQIE10.label() + " " +
                  _tagQIE10.instance());

  double HADM_sum = 0, HADP_sum = 0, EMM_sum = 0, EMP_sum = 0;
  double HADM_tot_sum = 0, HADP_tot_sum = 0, EMM_tot_sum = 0, EMP_tot_sum = 0;

  for (auto it = digis->begin(); it != digis->end(); it++) {
    const QIE10DataFrame digi = static_cast<const QIE10DataFrame>(*it);

    HcalZDCDetId const& did = digi.detid();
    bool isFSC = (did.zside() < 0 && did.section() == 1 && did.channel() > 6);

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
      // ZDC Minus (exclude FSC)
      else if (!isFSC) {
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
        // fill the tdc time the same way as in the reco
        float tmp_tdctime = 0;
        // TDC error codes will be 60=-1, 61 = -2, 62 = -3, 63 = -4
        // assume max amplitude should occur in TS2
        if (digi[i].le_tdc() >= 60)
          tmp_tdctime = -1 * (digi[i].le_tdc() - 59);
        else
          tmp_tdctime = 50. + (digi[i].le_tdc() / 2);
        _cTDC_EChannel[did()]->Fill(tmp_tdctime);
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
