#include <string>

#include "DQM/L1TMonitor/interface/L1TdeCSCTPG.h"

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
      B904Setup_(ps.getParameter<bool>("B904Setup")),
      isRun3_(ps.getParameter<bool>("isRun3")),
      preTriggerAnalysis_(ps.getParameter<bool>("preTriggerAnalysis")) {}

L1TdeCSCTPG::~L1TdeCSCTPG() {}

void L1TdeCSCTPG::bookHistograms(DQMStore::IBooker& iBooker, const edm::Run&, const edm::EventSetup&) {
  iBooker.setCurrentFolder(monitorDir_);

  // do not analyze Run-3 properties in Run-1 and Run-2 eras
  if (!isRun3_) {
    clctVars_.resize(4);
    lctVars_.resize(5);
  }

  // remove the non-ME1/1 chambers from the list when B904Setup is set to true
  if (B904Setup_) {
    chambers_.resize(1);
  }
  // do not analyze the 1/4-strip bit, 1/8-strip bit
  else {
    clctVars_.resize(9);
    lctVars_.resize(9);
  }

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
  if (B904Setup_)
    e.getByToken(emulpreCLCT_token_, emulpreCLCTs);

  for (auto it = dataALCTs->begin(); it != dataALCTs->end(); it++) {
    auto range = dataALCTs->get((*it).first);
    const int type = ((*it).first).iChamberType() - 2;
    // ignore non-ME1/1 chambers when using B904 test-stand data
    if (B904Setup_ and !((*it).first).isME11())
      continue;
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
    const int type = ((*it).first).iChamberType() - 2;
    // ignore non-ME1/1 chambers when using B904 test-stand data
    if (B904Setup_ and !((*it).first).isME11())
      continue;
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
    const int type = ((*it).first).iChamberType() - 2;
    // ignore non-ME1/1 chambers when using B904 test-stand data
    if (B904Setup_ and !((*it).first).isME11())
      continue;
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
          if (B904Setup_) {
            chamberHistos[type]["clct_quartstripbit_data"]->Fill(clct->getQuartStripBit());
            chamberHistos[type]["clct_eighthstripbit_data"]->Fill(clct->getEighthStripBit());
          }
        }
      }
    }
  }

  for (auto it = emulCLCTs->begin(); it != emulCLCTs->end(); it++) {
    auto range = emulCLCTs->get((*it).first);
    const int type = ((*it).first).iChamberType() - 2;
    // ignore non-ME1/1 chambers when using B904 test-stand data
    if (B904Setup_ and !((*it).first).isME11())
      continue;
    // remove the duplicate CLCTs
    // these are CLCTs that have the same properties as CLCTs found
    // before by the emulator, except for the BX, which is off by +1
    std::vector<CSCCLCTDigi> cleanedemul;
    for (auto clct = range.first; clct != range.second; clct++) {
      if (not isDuplicateCLCT(*clct, cleanedemul))
        cleanedemul.push_back(*clct);
    }
    for (const auto& clct : cleanedemul) {
      if (clct.isValid()) {
        if (preTriggerAnalysis_) {
          tempemul.push_back(clct);
        }
        chamberHistos[type]["clct_pattern_emul"]->Fill(clct.getPattern());
        chamberHistos[type]["clct_quality_emul"]->Fill(clct.getQuality());
        chamberHistos[type]["clct_halfstrip_emul"]->Fill(clct.getKeyStrip());
        chamberHistos[type]["clct_bend_emul"]->Fill(clct.getBend());
        if (isRun3_) {
          chamberHistos[type]["clct_run3pattern_emul"]->Fill(clct.getRun3Pattern());
          chamberHistos[type]["clct_quartstrip_emul"]->Fill(clct.getKeyStrip(4));
          chamberHistos[type]["clct_eighthstrip_emul"]->Fill(clct.getKeyStrip(8));
          chamberHistos[type]["clct_slope_emul"]->Fill(clct.getSlope());
          chamberHistos[type]["clct_compcode_emul"]->Fill(clct.getCompCode());
          if (B904Setup_) {
            chamberHistos[type]["clct_quartstripbit_emul"]->Fill(clct.getQuartStripBit());
            chamberHistos[type]["clct_eighthstripbit_emul"]->Fill(clct.getEighthStripBit());
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
    const int type = ((*it).first).iChamberType() - 2;
    // ignore non-ME1/1 chambers when using B904 test-stand data
    if (B904Setup_ and !((*it).first).isME11())
      continue;
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
          if (B904Setup_) {
            chamberHistos[type]["lct_quartstripbit_data"]->Fill(lct->getQuartStripBit());
            chamberHistos[type]["lct_eighthstripbit_data"]->Fill(lct->getEighthStripBit());
          }
        }
      }
    }
  }

  for (auto it = emulLCTs->begin(); it != emulLCTs->end(); it++) {
    auto range = emulLCTs->get((*it).first);
    const int type = ((*it).first).iChamberType() - 2;
    // ignore non-ME1/1 chambers when using B904 test-stand data
    if (B904Setup_ and !((*it).first).isME11())
      continue;

    // remove the duplicate LCTs
    // these are LCTs that have the same properties as LCTs found
    // before by the emulator, except for the BX, which is off by +1
    std::vector<CSCCorrelatedLCTDigi> cleanedemul;
    for (auto lct = range.first; lct != range.second; lct++) {
      if (not isDuplicateLCT(*lct, cleanedemul))
        cleanedemul.push_back(*lct);
    }

    for (const auto& lct : cleanedemul) {
      if (lct.isValid()) {
        chamberHistos[type]["lct_pattern_emul"]->Fill(lct.getPattern());
        chamberHistos[type]["lct_quality_emul"]->Fill(lct.getQuality());
        chamberHistos[type]["lct_wiregroup_emul"]->Fill(lct.getKeyWG());
        chamberHistos[type]["lct_halfstrip_emul"]->Fill(lct.getStrip());
        chamberHistos[type]["lct_bend_emul"]->Fill(lct.getBend());
        if (isRun3_) {
          chamberHistos[type]["lct_run3pattern_emul"]->Fill(lct.getRun3Pattern());
          chamberHistos[type]["lct_slope_emul"]->Fill(lct.getSlope());
          chamberHistos[type]["lct_quartstrip_emul"]->Fill(lct.getStrip(4));
          chamberHistos[type]["lct_eighthstrip_emul"]->Fill(lct.getStrip(8));
          if (B904Setup_) {
            chamberHistos[type]["lct_quartstripbit_emul"]->Fill(lct.getQuartStripBit());
            chamberHistos[type]["lct_eighthstripbit_emul"]->Fill(lct.getEighthStripBit());
          }
        }
      }
    }
  }
}

bool L1TdeCSCTPG::isDuplicateCLCT(const CSCCLCTDigi& clct, const std::vector<CSCCLCTDigi>& container) const {
  // if the temporary container is empty, the TP cannot be a duplicate
  if (container.empty())
    return false;
  else {
    for (const auto& rhs : container) {
      if (isCLCTOffByOneBX(clct, rhs))
        return true;
    }
    return false;
  }
}

bool L1TdeCSCTPG::isDuplicateLCT(const CSCCorrelatedLCTDigi& lct,
                                 const std::vector<CSCCorrelatedLCTDigi>& container) const {
  // if the temporary container is empty, the TP cannot be a duplicate
  if (container.empty())
    return false;
  else {
    for (const auto& rhs : container) {
      if (isLCTOffByOneBX(lct, rhs))
        return true;
    }
    return false;
  }
}

bool L1TdeCSCTPG::isCLCTOffByOneBX(const CSCCLCTDigi& lhs, const CSCCLCTDigi& rhs) const {
  // because the comparator code is degenerate (several comparator codes can produce the
  // same slope and position), we leave it out of the comparison
  bool returnValue = false;
  if (lhs.isValid() == rhs.isValid() && lhs.getQuality() == rhs.getQuality() && lhs.getPattern() == rhs.getPattern() &&
      lhs.getRun3Pattern() == rhs.getRun3Pattern() && lhs.getKeyStrip() == rhs.getKeyStrip() &&
      lhs.getStripType() == rhs.getStripType() && lhs.getBend() == rhs.getBend() && lhs.getBX() == rhs.getBX() + 1 &&
      lhs.getQuartStripBit() == rhs.getQuartStripBit() && lhs.getEighthStripBit() == rhs.getEighthStripBit()) {
    returnValue = true;
  }
  return returnValue;
}

bool L1TdeCSCTPG::isLCTOffByOneBX(const CSCCorrelatedLCTDigi& lhs, const CSCCorrelatedLCTDigi& rhs) const {
  bool returnValue = false;
  if (lhs.isValid() == rhs.isValid() && lhs.getQuality() == rhs.getQuality() && lhs.getPattern() == rhs.getPattern() &&
      lhs.getRun3Pattern() == rhs.getRun3Pattern() && lhs.getStrip() == rhs.getStrip() &&
      lhs.getStripType() == rhs.getStripType() && lhs.getBend() == rhs.getBend() && lhs.getBX() == rhs.getBX() + 1 &&
      lhs.getQuartStripBit() == rhs.getQuartStripBit() && lhs.getEighthStripBit() == rhs.getEighthStripBit() &&
      lhs.getKeyWG() == rhs.getKeyWG()) {
    returnValue = true;
  }
  return returnValue;
}
