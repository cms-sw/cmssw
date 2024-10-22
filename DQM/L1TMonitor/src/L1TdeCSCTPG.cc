#include <string>

#include "DQM/L1TMonitor/interface/L1TdeCSCTPG.h"
#include "DataFormats/MuonDetId/interface/CSCTriggerNumbering.h"

L1TdeCSCTPG::L1TdeCSCTPG(const edm::ParameterSet& ps)
    : dataALCT_token_(consumes<CSCALCTDigiCollection>(ps.getParameter<edm::InputTag>("dataALCT"))),
      emulALCT_token_(consumes<CSCALCTDigiCollection>(ps.getParameter<edm::InputTag>("emulALCT"))),
      dataCLCT_token_(consumes<CSCCLCTDigiCollection>(ps.getParameter<edm::InputTag>("dataCLCT"))),
      emulCLCT_token_(consumes<CSCCLCTDigiCollection>(ps.getParameter<edm::InputTag>("emulCLCT"))),
      emulpreCLCT_token_(consumes<CSCCLCTPreTriggerDigiCollection>(ps.getParameter<edm::InputTag>("emulpreCLCT"))),
      dataLCT_token_(consumes<CSCCorrelatedLCTDigiCollection>(ps.getParameter<edm::InputTag>("dataLCT"))),
      emulLCT_token_(consumes<CSCCorrelatedLCTDigiCollection>(ps.getParameter<edm::InputTag>("emulLCT"))),
      monitorDir_(ps.getParameter<std::string>("monitorDir")),

      chambers_(ps.getParameter<std::vector<std::string>>("chambers")),
      dataEmul_(ps.getParameter<std::vector<std::string>>("dataEmul")),

      // variables
      alctVars_(ps.getParameter<std::vector<std::string>>("alctVars")),
      clctVars_(ps.getParameter<std::vector<std::string>>("clctVars")),
      lctVars_(ps.getParameter<std::vector<std::string>>("lctVars")),

      // binning
      alctNBin_(ps.getParameter<std::vector<unsigned>>("alctNBin")),
      clctNBin_(ps.getParameter<std::vector<unsigned>>("clctNBin")),
      lctNBin_(ps.getParameter<std::vector<unsigned>>("lctNBin")),
      alctMinBin_(ps.getParameter<std::vector<double>>("alctMinBin")),
      clctMinBin_(ps.getParameter<std::vector<double>>("clctMinBin")),
      lctMinBin_(ps.getParameter<std::vector<double>>("lctMinBin")),
      alctMaxBin_(ps.getParameter<std::vector<double>>("alctMaxBin")),
      clctMaxBin_(ps.getParameter<std::vector<double>>("clctMaxBin")),
      lctMaxBin_(ps.getParameter<std::vector<double>>("lctMaxBin")),
      // options for test stands at B904
      useB904ME11_(ps.getParameter<bool>("useB904ME11")),
      useB904ME21_(ps.getParameter<bool>("useB904ME21")),
      useB904ME234s2_(ps.getParameter<bool>("useB904ME234s2")),
      isRun3_(ps.getParameter<bool>("isRun3")),
      make1DPlots_(ps.getParameter<bool>("make1DPlots")),
      preTriggerAnalysis_(ps.getParameter<bool>("preTriggerAnalysis")) {
  useB904_ = useB904ME11_ or useB904ME21_ or useB904ME234s2_;
}

L1TdeCSCTPG::~L1TdeCSCTPG() {}

void L1TdeCSCTPG::bookHistograms(DQMStore::IBooker& iBooker, const edm::Run&, const edm::EventSetup&) {
  iBooker.setCurrentFolder(monitorDir_);

  // remove the non-ME1/1 chambers from the list when useB904ME11 is set to true
  if (useB904ME11_) {
    chambers_.resize(1);
  }
  // similar for ME2/1
  else if (useB904ME21_) {
    auto temp = chambers_[3];
    chambers_.resize(1);
    chambers_[0] = temp;
  }
  // similar for ME4/2
  else if (useB904ME234s2_) {
    auto temp = chambers_.back();
    chambers_.resize(1);
    chambers_[0] = temp;
  }
  // collision data in Run-3
  else if (isRun3_) {
    clctVars_.resize(9);
    lctVars_.resize(9);
  }
  // do not analyze Run-3 properties in Run-1 and Run-2 eras
  else {
    clctVars_.resize(4);
    lctVars_.resize(5);
  }

  // 1D plots for experts
  if (useB904ME11_ or useB904ME21_ or useB904ME234s2_ or make1DPlots_) {
    // chamber type
    for (unsigned iType = 0; iType < chambers_.size(); iType++) {
      // data vs emulator
      for (unsigned iData = 0; iData < dataEmul_.size(); iData++) {
        // alct variable
        for (unsigned iVar = 0; iVar < alctVars_.size(); iVar++) {
          const std::string key("alct_" + alctVars_[iVar] + "_" + dataEmul_[iData]);
          const std::string histName(key + "_" + chambers_[iType]);
          const std::string histTitle(chambers_[iType] + " ALCT " + alctVars_[iVar] + " (" + dataEmul_[iData] + ") ");
          chamberHistos[iType][key] =
              iBooker.book1D(histName, histTitle, alctNBin_[iVar], alctMinBin_[iVar], alctMaxBin_[iVar]);
          chamberHistos[iType][key]->getTH1()->SetMinimum(0);
        }

        // clct variable
        for (unsigned iVar = 0; iVar < clctVars_.size(); iVar++) {
          const std::string key("clct_" + clctVars_[iVar] + "_" + dataEmul_[iData]);
          const std::string histName(key + "_" + chambers_[iType]);
          const std::string histTitle(chambers_[iType] + " CLCT " + clctVars_[iVar] + " (" + dataEmul_[iData] + ") ");
          chamberHistos[iType][key] =
              iBooker.book1D(histName, histTitle, clctNBin_[iVar], clctMinBin_[iVar], clctMaxBin_[iVar]);
          chamberHistos[iType][key]->getTH1()->SetMinimum(0);
        }

        // lct variable
        for (unsigned iVar = 0; iVar < lctVars_.size(); iVar++) {
          const std::string key("lct_" + lctVars_[iVar] + "_" + dataEmul_[iData]);
          const std::string histName(key + "_" + chambers_[iType]);
          const std::string histTitle(chambers_[iType] + " LCT " + lctVars_[iVar] + " (" + dataEmul_[iData] + ") ");
          chamberHistos[iType][key] =
              iBooker.book1D(histName, histTitle, lctNBin_[iVar], lctMinBin_[iVar], lctMaxBin_[iVar]);
          chamberHistos[iType][key]->getTH1()->SetMinimum(0);
        }
      }
    }
  }

  // 2D summary plots

  // These plots are for showing the efficiency that the unpacked data are being correctly emulated (having a emulated data counterpart)
  lctDataSummary_denom_ = iBooker.book2D("lct_csctp_data_summary_denom", "LCT Summary", 36, 1, 37, 18, 0, 18);
  lctDataSummary_num_ = iBooker.book2D("lct_csctp_data_summary_num", "LCT Summary", 36, 1, 37, 18, 0, 18);
  alctDataSummary_denom_ = iBooker.book2D("alct_csctp_data_summary_denom", "ALCT Summary", 36, 1, 37, 18, 0, 18);
  alctDataSummary_num_ = iBooker.book2D("alct_csctp_data_summary_num", "ALCT Summary", 36, 1, 37, 18, 0, 18);
  clctDataSummary_denom_ = iBooker.book2D("clct_csctp_data_summary_denom", "CLCT Summary", 36, 1, 37, 18, 0, 18);
  clctDataSummary_num_ = iBooker.book2D("clct_csctp_data_summary_num", "CLCT Summary", 36, 1, 37, 18, 0, 18);

  // These plots are for showing the fraction of emulated data that does not have counterparts in the unpacked data
  lctEmulSummary_denom_ = iBooker.book2D("lct_csctp_emul_summary_denom", "LCT Summary", 36, 1, 37, 18, 0, 18);
  lctEmulSummary_num_ = iBooker.book2D("lct_csctp_emul_summary_num", "LCT Summary", 36, 1, 37, 18, 0, 18);
  alctEmulSummary_denom_ = iBooker.book2D("alct_csctp_emul_summary_denom", "ALCT Summary", 36, 1, 37, 18, 0, 18);
  alctEmulSummary_num_ = iBooker.book2D("alct_csctp_emul_summary_num", "ALCT Summary", 36, 1, 37, 18, 0, 18);
  clctEmulSummary_denom_ = iBooker.book2D("clct_csctp_emul_summary_denom", "CLCT Summary", 36, 1, 37, 18, 0, 18);
  clctEmulSummary_num_ = iBooker.book2D("clct_csctp_emul_summary_num", "CLCT Summary", 36, 1, 37, 18, 0, 18);

  // x labels
  lctDataSummary_denom_->setAxisTitle("Chamber", 1);
  lctDataSummary_num_->setAxisTitle("Chamber", 1);
  alctDataSummary_denom_->setAxisTitle("Chamber", 1);
  alctDataSummary_num_->setAxisTitle("Chamber", 1);
  clctDataSummary_denom_->setAxisTitle("Chamber", 1);
  clctDataSummary_num_->setAxisTitle("Chamber", 1);

  lctEmulSummary_denom_->setAxisTitle("Chamber", 1);
  lctEmulSummary_num_->setAxisTitle("Chamber", 1);
  alctEmulSummary_denom_->setAxisTitle("Chamber", 1);
  alctEmulSummary_num_->setAxisTitle("Chamber", 1);
  clctEmulSummary_denom_->setAxisTitle("Chamber", 1);
  clctEmulSummary_num_->setAxisTitle("Chamber", 1);

  // plotting option
  lctDataSummary_denom_->setOption("colz");
  lctDataSummary_num_->setOption("colz");
  alctDataSummary_denom_->setOption("colz");
  alctDataSummary_num_->setOption("colz");
  clctDataSummary_denom_->setOption("colz");
  clctDataSummary_num_->setOption("colz");

  lctEmulSummary_denom_->setOption("colz");
  lctEmulSummary_num_->setOption("colz");
  alctEmulSummary_denom_->setOption("colz");
  alctEmulSummary_num_->setOption("colz");
  clctEmulSummary_denom_->setOption("colz");
  clctEmulSummary_num_->setOption("colz");

  // summary plots
  const std::array<std::string, 9> suffix_label{{"4/2", "4/1", "3/2", "3/1", " 2/2", "2/1", "1/3", "1/2", "1/1"}};

  // y labels
  for (int ybin = 1; ybin <= 9; ++ybin) {
    lctDataSummary_denom_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    lctDataSummary_num_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    alctDataSummary_denom_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    alctDataSummary_num_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    clctDataSummary_denom_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    clctDataSummary_num_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);

    lctEmulSummary_denom_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    lctEmulSummary_num_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    alctEmulSummary_denom_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    alctEmulSummary_num_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    clctEmulSummary_denom_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    clctEmulSummary_num_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);

    lctDataSummary_denom_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    lctDataSummary_num_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    alctDataSummary_denom_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    alctDataSummary_num_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    clctDataSummary_denom_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    clctDataSummary_num_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);

    lctEmulSummary_denom_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    lctEmulSummary_num_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    alctEmulSummary_denom_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    alctEmulSummary_num_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    clctEmulSummary_denom_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    clctEmulSummary_num_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
  }
}

void L1TdeCSCTPG::analyze(const edm::Event& e, const edm::EventSetup& c) {
  // handles
  edm::Handle<CSCALCTDigiCollection> dataALCTs;
  edm::Handle<CSCALCTDigiCollection> emulALCTs;
  edm::Handle<CSCCLCTDigiCollection> dataCLCTs;
  edm::Handle<CSCCLCTDigiCollection> emulCLCTs;
  edm::Handle<CSCCLCTPreTriggerDigiCollection> emulpreCLCTs;
  edm::Handle<CSCCorrelatedLCTDigiCollection> dataLCTs;
  edm::Handle<CSCCorrelatedLCTDigiCollection> emulLCTs;

  e.getByToken(dataALCT_token_, dataALCTs);
  e.getByToken(emulALCT_token_, emulALCTs);
  e.getByToken(dataCLCT_token_, dataCLCTs);
  e.getByToken(emulCLCT_token_, emulCLCTs);
  e.getByToken(dataLCT_token_, dataLCTs);
  e.getByToken(emulLCT_token_, emulLCTs);
  // only do pre-trigger analysis when B904 setup is used
  if (useB904_)
    e.getByToken(emulpreCLCT_token_, emulpreCLCTs);

  // 1D plots for experts
  if (useB904ME11_ or useB904ME21_ or useB904ME234s2_ or make1DPlots_) {
    for (auto it = dataALCTs->begin(); it != dataALCTs->end(); it++) {
      auto range = dataALCTs->get((*it).first);
      const CSCDetId& detid((*it).first);
      int type = ((*it).first).iChamberType() - 2;
      // ignore non-ME1/1 chambers when using B904 test-stand data
      if (useB904ME11_ and !(detid.isME11()))
        continue;
      if (useB904ME21_ and !(detid.isME21()))
        continue;
      if (useB904ME234s2_ and !(detid.isME22() or detid.isME32() or detid.isME42()))
        continue;
      // to prevent crashes because you are booking histos for single b904 chamber
      if (useB904ME234s2_ or useB904ME21_)
        type = 0;
      for (auto alct = range.first; alct != range.second; alct++) {
        if (alct->isValid()) {
          chamberHistos[type]["alct_quality_data"]->Fill(alct->getQuality());
          chamberHistos[type]["alct_wiregroup_data"]->Fill(alct->getKeyWG());
          chamberHistos[type]["alct_bx_data"]->Fill(alct->getBX());
        }
      }
    }

    for (auto it = emulALCTs->begin(); it != emulALCTs->end(); it++) {
      auto range = emulALCTs->get((*it).first);
      const CSCDetId& detid((*it).first);
      int type = ((*it).first).iChamberType() - 2;
      // ignore non-ME1/1 chambers when using B904 test-stand data
      if (useB904ME11_ and !(detid.isME11()))
        continue;
      if (useB904ME21_ and !(detid.isME21()))
        continue;
      if (useB904ME234s2_ and !(detid.isME22() or detid.isME32() or detid.isME42()))
        continue;
      // to prevent crashes because you are booking histos for single b904 chamber
      if (useB904ME234s2_ or useB904ME21_)
        type = 0;
      for (auto alct = range.first; alct != range.second; alct++) {
        if (alct->isValid()) {
          chamberHistos[type]["alct_quality_emul"]->Fill(alct->getQuality());
          chamberHistos[type]["alct_wiregroup_emul"]->Fill(alct->getKeyWG());
          chamberHistos[type]["alct_bx_emul"]->Fill(alct->getBX());
        }
      }
    }

    // temporary containers for B904 analysis
    std::vector<CSCCLCTDigi> tempdata;
    std::vector<CSCCLCTDigi> tempemul;

    for (auto it = dataCLCTs->begin(); it != dataCLCTs->end(); it++) {
      auto range = dataCLCTs->get((*it).first);
      const CSCDetId& detid((*it).first);
      int type = ((*it).first).iChamberType() - 2;
      // ignore non-ME1/1 chambers when using B904 test-stand data
      if (useB904ME11_ and !(detid.isME11()))
        continue;
      if (useB904ME21_ and !(detid.isME21()))
        continue;
      if (useB904ME234s2_ and !(detid.isME22() or detid.isME32() or detid.isME42()))
        continue;
      // to prevent crashes because you are booking histos for single b904 chamber
      if (useB904ME234s2_ or useB904ME21_)
        type = 0;
      for (auto clct = range.first; clct != range.second; clct++) {
        if (clct->isValid()) {
          if (preTriggerAnalysis_) {
            tempdata.push_back(*clct);
          }
          chamberHistos[type]["clct_pattern_data"]->Fill(clct->getPattern());
          chamberHistos[type]["clct_quality_data"]->Fill(clct->getQuality());
          chamberHistos[type]["clct_halfstrip_data"]->Fill(clct->getKeyStrip());
          chamberHistos[type]["clct_bend_data"]->Fill(clct->getBend());
          if (isRun3_) {
            chamberHistos[type]["clct_run3pattern_data"]->Fill(clct->getRun3Pattern());
            chamberHistos[type]["clct_quartstrip_data"]->Fill(clct->getKeyStrip(4));
            chamberHistos[type]["clct_eighthstrip_data"]->Fill(clct->getKeyStrip(8));
            chamberHistos[type]["clct_slope_data"]->Fill(clct->getSlope());
            chamberHistos[type]["clct_compcode_data"]->Fill(clct->getCompCode());
            if (useB904_) {
              chamberHistos[type]["clct_quartstripbit_data"]->Fill(clct->getQuartStripBit());
              chamberHistos[type]["clct_eighthstripbit_data"]->Fill(clct->getEighthStripBit());
            }
          }
        }
      }
    }

    for (auto it = emulCLCTs->begin(); it != emulCLCTs->end(); it++) {
      auto range = emulCLCTs->get((*it).first);
      const CSCDetId& detid((*it).first);
      int type = ((*it).first).iChamberType() - 2;
      // ignore non-ME1/1 chambers when using B904 test-stand data
      if (useB904ME11_ and !(detid.isME11()))
        continue;
      if (useB904ME21_ and !(detid.isME21()))
        continue;
      if (useB904ME234s2_ and !(detid.isME22() or detid.isME32() or detid.isME42()))
        continue;
      // to prevent crashes because you are booking histos for single b904 chamber
      if (useB904ME234s2_ or useB904ME21_)
        type = 0;
      for (auto clct = range.first; clct != range.second; clct++) {
        if (clct->isValid()) {
          if (preTriggerAnalysis_) {
            tempemul.push_back(*clct);
          }
          chamberHistos[type]["clct_pattern_emul"]->Fill(clct->getPattern());
          chamberHistos[type]["clct_quality_emul"]->Fill(clct->getQuality());
          chamberHistos[type]["clct_halfstrip_emul"]->Fill(clct->getKeyStrip());
          chamberHistos[type]["clct_bend_emul"]->Fill(clct->getBend());
          if (isRun3_) {
            chamberHistos[type]["clct_run3pattern_emul"]->Fill(clct->getRun3Pattern());
            chamberHistos[type]["clct_quartstrip_emul"]->Fill(clct->getKeyStrip(4));
            chamberHistos[type]["clct_eighthstrip_emul"]->Fill(clct->getKeyStrip(8));
            chamberHistos[type]["clct_slope_emul"]->Fill(clct->getSlope());
            chamberHistos[type]["clct_compcode_emul"]->Fill(clct->getCompCode());
            if (useB904_) {
              chamberHistos[type]["clct_quartstripbit_emul"]->Fill(clct->getQuartStripBit());
              chamberHistos[type]["clct_eighthstripbit_emul"]->Fill(clct->getEighthStripBit());
            }
          }
        }
      }
    }

    // Pre-trigger analysis
    if (preTriggerAnalysis_) {
      if (tempdata.size() != tempemul.size()) {
        for (auto& clct : tempdata) {
          edm::LogWarning("L1TdeCSCTPG") << "data" << clct;
        }
        for (auto& clct : tempemul) {
          edm::LogWarning("L1TdeCSCTPG") << "emul" << clct;
        }
        for (auto it = emulpreCLCTs->begin(); it != emulpreCLCTs->end(); it++) {
          auto range = emulpreCLCTs->get((*it).first);
          for (auto clct = range.first; clct != range.second; clct++) {
            edm::LogWarning("L1TdeCSCTPG") << "emul pre" << *clct;
          }
        }
      }
    }

    for (auto it = dataLCTs->begin(); it != dataLCTs->end(); it++) {
      auto range = dataLCTs->get((*it).first);
      const CSCDetId& detid((*it).first);
      int type = ((*it).first).iChamberType() - 2;
      // ignore non-ME1/1 chambers when using B904 test-stand data
      if (useB904ME11_ and !(detid.isME11()))
        continue;
      if (useB904ME21_ and !(detid.isME21()))
        continue;
      if (useB904ME234s2_ and !(detid.isME22() or detid.isME32() or detid.isME42()))
        continue;
      // to prevent crashes because you are booking histos for single b904 chamber
      if (useB904ME234s2_ or useB904ME21_)
        type = 0;
      for (auto lct = range.first; lct != range.second; lct++) {
        if (lct->isValid()) {
          chamberHistos[type]["lct_pattern_data"]->Fill(lct->getPattern());
          chamberHistos[type]["lct_quality_data"]->Fill(lct->getQuality());
          chamberHistos[type]["lct_wiregroup_data"]->Fill(lct->getKeyWG());
          chamberHistos[type]["lct_halfstrip_data"]->Fill(lct->getStrip());
          chamberHistos[type]["lct_bend_data"]->Fill(lct->getBend());
          if (isRun3_) {
            chamberHistos[type]["lct_run3pattern_data"]->Fill(lct->getRun3Pattern());
            chamberHistos[type]["lct_slope_data"]->Fill(lct->getSlope());
            chamberHistos[type]["lct_quartstrip_data"]->Fill(lct->getStrip(4));
            chamberHistos[type]["lct_eighthstrip_data"]->Fill(lct->getStrip(8));
            if (useB904_) {
              chamberHistos[type]["lct_quartstripbit_data"]->Fill(lct->getQuartStripBit());
              chamberHistos[type]["lct_eighthstripbit_data"]->Fill(lct->getEighthStripBit());
            }
          }
        }
      }
    }

    for (auto it = emulLCTs->begin(); it != emulLCTs->end(); it++) {
      auto range = emulLCTs->get((*it).first);
      const CSCDetId& detid((*it).first);
      int type = ((*it).first).iChamberType() - 2;
      // ignore non-ME1/1 chambers when using B904 test-stand data
      if (useB904ME11_ and !(detid.isME11()))
        continue;
      if (useB904ME21_ and !(detid.isME21()))
        continue;
      if (useB904ME234s2_ and !(detid.isME22() or detid.isME32() or detid.isME42()))
        continue;
      // to prevent crashes because you are booking histos for single b904 chamber
      if (useB904ME234s2_ or useB904ME21_)
        type = 0;
      for (auto lct = range.first; lct != range.second; lct++) {
        if (lct->isValid()) {
          chamberHistos[type]["lct_pattern_emul"]->Fill(lct->getPattern());
          chamberHistos[type]["lct_quality_emul"]->Fill(lct->getQuality());
          chamberHistos[type]["lct_wiregroup_emul"]->Fill(lct->getKeyWG());
          chamberHistos[type]["lct_halfstrip_emul"]->Fill(lct->getStrip());
          chamberHistos[type]["lct_bend_emul"]->Fill(lct->getBend());
          if (isRun3_) {
            chamberHistos[type]["lct_run3pattern_emul"]->Fill(lct->getRun3Pattern());
            chamberHistos[type]["lct_slope_emul"]->Fill(lct->getSlope());
            chamberHistos[type]["lct_quartstrip_emul"]->Fill(lct->getStrip(4));
            chamberHistos[type]["lct_eighthstrip_emul"]->Fill(lct->getStrip(8));
            if (useB904_) {
              chamberHistos[type]["lct_quartstripbit_emul"]->Fill(lct->getQuartStripBit());
              chamberHistos[type]["lct_eighthstripbit_emul"]->Fill(lct->getEighthStripBit());
            }
          }
        }
      }
    }
  }

  // summary plots
  const std::map<std::pair<int, int>, int> histIndexCSC = {{{1, 1}, 8},
                                                           {{1, 2}, 7},
                                                           {{1, 3}, 6},
                                                           {{2, 1}, 5},
                                                           {{2, 2}, 4},
                                                           {{3, 1}, 3},
                                                           {{3, 2}, 2},
                                                           {{4, 1}, 1},
                                                           {{4, 2}, 0}};

  const int min_endcap = CSCDetId::minEndcapId();
  const int max_endcap = CSCDetId::maxEndcapId();
  const int min_station = CSCDetId::minStationId();
  const int max_station = CSCDetId::maxStationId();
  const int min_sector = CSCTriggerNumbering::minTriggerSectorId();
  const int max_sector = CSCTriggerNumbering::maxTriggerSectorId();
  const int min_subsector = CSCTriggerNumbering::minTriggerSubSectorId();
  const int max_subsector = CSCTriggerNumbering::maxTriggerSubSectorId();
  const int min_chamber = CSCTriggerNumbering::minTriggerCscId();
  const int max_chamber = CSCTriggerNumbering::maxTriggerCscId();

  for (int endc = min_endcap; endc <= max_endcap; endc++) {
    // loop on all stations
    for (int stat = min_station; stat <= max_station; stat++) {
      int numsubs = ((stat == 1) ? max_subsector : 1);
      // loop on sectors and subsectors
      for (int sect = min_sector; sect <= max_sector; sect++) {
        for (int subs = min_subsector; subs <= numsubs; subs++) {
          // loop on all chambers
          for (int cham = min_chamber; cham <= max_chamber; cham++) {
            // extract the ring number
            int ring = CSCTriggerNumbering::ringFromTriggerLabels(stat, cham);

            // actual chamber number =/= trigger chamber number
            int chid = CSCTriggerNumbering::chamberFromTriggerLabels(sect, subs, stat, cham);

            // 0th layer means whole chamber.
            CSCDetId detid(endc, stat, ring, chid, 0);

            int chamber = detid.chamber();

            int sr = histIndexCSC.at({stat, ring});
            if (endc == 1)
              sr = 17 - sr;

            // ALCT analysis
            auto range_dataALCT = dataALCTs->get(detid);
            auto range_emulALCT = emulALCTs->get(detid);

            for (auto dalct = range_dataALCT.first; dalct != range_dataALCT.second; dalct++) {
              if (dalct->isValid()) {
                alctDataSummary_denom_->Fill(chamber, sr);
                // check for least one matching ALCT
                for (auto ealct = range_emulALCT.first; ealct != range_emulALCT.second; ealct++) {
                  if (ealct->isValid() and *dalct == *ealct) {
                    alctDataSummary_num_->Fill(chamber, sr);
                  }
                }
              }
            }

            for (auto ealct = range_emulALCT.first; ealct != range_emulALCT.second; ealct++) {
              bool isMatched = false;
              if (ealct->isValid()) {
                alctEmulSummary_denom_->Fill(chamber, sr);
                // check for least one matching ALCT
                for (auto dalct = range_dataALCT.first; dalct != range_dataALCT.second; dalct++) {
                  if (*dalct == *ealct)
                    isMatched = true;
                }
                // only fill when it is not matched to an ALCT
                // to understand if the emulator is producing too many ALCTs
                if (!isMatched) {
                  alctEmulSummary_num_->Fill(chamber, sr);
                }
              }
            }

            // CLCT analysis
            auto range_dataCLCT = dataCLCTs->get(detid);
            auto range_emulCLCT = emulCLCTs->get(detid);

            for (auto dclct = range_dataCLCT.first; dclct != range_dataCLCT.second; dclct++) {
              if (dclct->isValid()) {
                clctDataSummary_denom_->Fill(chamber, sr);
                // check for least one matching CLCT
                for (auto eclct = range_emulCLCT.first; eclct != range_emulCLCT.second; eclct++) {
                  if (eclct->isValid() and areSameCLCTs(*dclct, *eclct)) {
                    clctDataSummary_num_->Fill(chamber, sr);
                  }
                }
              }
            }

            for (auto eclct = range_emulCLCT.first; eclct != range_emulCLCT.second; eclct++) {
              bool isMatched = false;
              if (eclct->isValid()) {
                clctEmulSummary_denom_->Fill(chamber, sr);
                // check for least one matching CLCT
                for (auto dclct = range_dataCLCT.first; dclct != range_dataCLCT.second; dclct++) {
                  if (areSameCLCTs(*dclct, *eclct))
                    isMatched = true;
                }
                // only fill when it is not matched to an CLCT
                // to understand if the emulator is producing too many CLCTs
                if (!isMatched) {
                  clctEmulSummary_num_->Fill(chamber, sr);
                }
              }
            }

            // LCT analysis
            auto range_dataLCT = dataLCTs->get(detid);
            auto range_emulLCT = emulLCTs->get(detid);

            for (auto dlct = range_dataLCT.first; dlct != range_dataLCT.second; dlct++) {
              if (dlct->isValid()) {
                lctDataSummary_denom_->Fill(chamber, sr);
                // check for least one matching LCT
                for (auto elct = range_emulLCT.first; elct != range_emulLCT.second; elct++) {
                  if (elct->isValid() and areSameLCTs(*dlct, *elct)) {
                    lctDataSummary_num_->Fill(chamber, sr);
                  }
                }
              }
            }

            for (auto elct = range_emulLCT.first; elct != range_emulLCT.second; elct++) {
              bool isMatched = false;
              if (elct->isValid()) {
                lctEmulSummary_denom_->Fill(chamber, sr);
                // check for least one matching LCT
                for (auto dlct = range_dataLCT.first; dlct != range_dataLCT.second; dlct++) {
                  if (areSameLCTs(*dlct, *elct))
                    isMatched = true;
                }
                // only fill when it is not matched to an LCT
                // to understand if the emulator is producing too many LCTs
                if (!isMatched) {
                  lctEmulSummary_num_->Fill(chamber, sr);
                }
              }
            }
          }
        }
      }
    }
  }
}

bool L1TdeCSCTPG::areSameCLCTs(const CSCCLCTDigi& lhs, const CSCCLCTDigi& rhs) const {
  // because the comparator code is degenerate (several comparator codes can produce the
  // same slope and position), we leave it out of the comparison
  // do not include the BX
  bool returnValue = false;
  if (lhs.isValid() == rhs.isValid() && lhs.getQuality() == rhs.getQuality() && lhs.getPattern() == rhs.getPattern() &&
      lhs.getRun3Pattern() == rhs.getRun3Pattern() && lhs.getKeyStrip() == rhs.getKeyStrip() &&
      lhs.getStripType() == rhs.getStripType() && lhs.getBend() == rhs.getBend() &&
      lhs.getQuartStripBit() == rhs.getQuartStripBit() && lhs.getEighthStripBit() == rhs.getEighthStripBit()) {
    returnValue = true;
  }
  return returnValue;
}

bool L1TdeCSCTPG::areSameLCTs(const CSCCorrelatedLCTDigi& lhs, const CSCCorrelatedLCTDigi& rhs) const {
  // do not include the BX
  bool returnValue = false;
  if (lhs.isValid() == rhs.isValid() && lhs.getQuality() == rhs.getQuality() && lhs.getPattern() == rhs.getPattern() &&
      lhs.getRun3Pattern() == rhs.getRun3Pattern() && lhs.getStrip() == rhs.getStrip() &&
      lhs.getStripType() == rhs.getStripType() && lhs.getBend() == rhs.getBend() &&
      lhs.getQuartStripBit() == rhs.getQuartStripBit() && lhs.getEighthStripBit() == rhs.getEighthStripBit() &&
      lhs.getKeyWG() == rhs.getKeyWG()) {
    returnValue = true;
  }
  return returnValue;
}
