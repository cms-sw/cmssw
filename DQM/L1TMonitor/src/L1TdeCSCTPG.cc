#include <string>

#include "DQM/L1TMonitor/interface/L1TdeCSCTPG.h"

L1TdeCSCTPG::L1TdeCSCTPG(const edm::ParameterSet& ps)
    : dataALCT_token_(consumes<CSCALCTDigiCollection>(ps.getParameter<edm::InputTag>("dataALCT"))),
      emulALCT_token_(consumes<CSCALCTDigiCollection>(ps.getParameter<edm::InputTag>("emulALCT"))),
      dataCLCT_token_(consumes<CSCCLCTDigiCollection>(ps.getParameter<edm::InputTag>("dataCLCT"))),
      emulCLCT_token_(consumes<CSCCLCTDigiCollection>(ps.getParameter<edm::InputTag>("emulCLCT"))),
      dataLCT_token_(consumes<CSCCorrelatedLCTDigiCollection>(ps.getParameter<edm::InputTag>("dataLCT"))),
      emulLCT_token_(consumes<CSCCorrelatedLCTDigiCollection>(ps.getParameter<edm::InputTag>("emulLCT"))),
      monitorDir_(ps.getParameter<std::string>("monitorDir")),
      verbose_(ps.getParameter<bool>("verbose")) {}

L1TdeCSCTPG::~L1TdeCSCTPG() {}

void L1TdeCSCTPG::bookHistograms(DQMStore::IBooker& ibooker, const edm::Run&, const edm::EventSetup&) {
  ibooker.setCurrentFolder(monitorDir_);

  for (unsigned iType = 0; iType < 10; iType++) {
    alct_quality_emul_[iType] = ibooker.book1D("CSC ALCT quality", "CSC ALCT quality", 16, 0.5, 16.5);
    alct_wiregroup_emul_[iType] = ibooker.book1D("CSC ALCT wire group", "CSC ALCT wire group", 116, -0.5, 115.5);
    alct_bx_emul_[iType] = ibooker.book1D("CSC ALCT bx", "CSC ALCT bx", 20, -0.5, 19.5);

    alct_quality_data_[iType] = ibooker.book1D("CSC ALCT quality", "CSC ALCT quality", 16, 0.5, 16.5);
    alct_wiregroup_data_[iType] = ibooker.book1D("CSC ALCT wire group", "CSC ALCT wire group", 116, -0.5, 115.5);
    alct_bx_data_[iType] = ibooker.book1D("CSC ALCT bx", "CSC ALCT bx", 20, -0.5, 19.5);

    clct_pattern_emul_[iType] = ibooker.book1D("CSC CLCT hit pattern", "CSC CLCT hit pattern", 16, -0.5, 15.5);
    clct_quality_emul_[iType] = ibooker.book1D("CSC CLCT quality", "CSC CLCT quality", 16, 0.5, 16.5);
    clct_halfstrip_emul_[iType] = ibooker.book1D("CSC CLCT halfstrip", "CSC CLCT strip", 224, -0.5, 223.5);
    clct_bend_emul_[iType] = ibooker.book1D("CSC CLCT bend", "CSC CLCT bend", 3, 0.5, 2.5);
    clct_bx_emul_[iType] = ibooker.book1D("CSC CLCT bx", "CSC CLCT bx", 20, -0.5, 19.5);

    clct_pattern_data_[iType] = ibooker.book1D("CSC CLCT hit pattern", "CSC CLCT hit pattern", 16, -0.5, 15.5);
    clct_quality_data_[iType] = ibooker.book1D("CSC CLCT quality", "CSC CLCT quality", 16, 0.5, 16.5);
    clct_halfstrip_data_[iType] = ibooker.book1D("CSC CLCT halfstrip", "CSC CLCT strip", 224, -0.5, 223.5);
    clct_bend_data_[iType] = ibooker.book1D("CSC CLCT bend", "CSC CLCT bend", 3, 0.5, 2.5);
    clct_bx_data_[iType] = ibooker.book1D("CSC CLCT bx", "CSC CLCT bx", 20, -0.5, 19.5);

    lct_pattern_emul_[iType] = ibooker.book1D("CSC LCT hit pattern", "CSC LCT hit pattern", 16, -0.5, 15.5);
    lct_quality_emul_[iType] = ibooker.book1D("CSC LCT quality", "CSC LCT quality", 16, 0.5, 16.5);
    lct_wiregroup_emul_[iType] = ibooker.book1D("CSC LCT wire group", "CSC LCT wire group", 116, -0.5, 115.5);
    lct_halfstrip_emul_[iType] = ibooker.book1D("CSC LCT halfstrip", "CSC LCT strip", 224, -0.5, 223.5);
    lct_bend_emul_[iType] = ibooker.book1D("CSC LCT bend", "CSC LCT bend", 3, 0.5, 2.5);
    lct_bx_emul_[iType] = ibooker.book1D("CSC LCT bx", "CSC LCT bx", 20, -0.5, 19.5);

    lct_pattern_data_[iType] = ibooker.book1D("CSC LCT hit pattern", "CSC LCT hit pattern", 16, -0.5, 15.5);
    lct_quality_data_[iType] = ibooker.book1D("CSC LCT quality", "CSC LCT quality", 16, 0.5, 16.5);
    lct_wiregroup_data_[iType] = ibooker.book1D("CSC LCT wire group", "CSC LCT wire group", 116, -0.5, 115.5);
    lct_halfstrip_data_[iType] = ibooker.book1D("CSC LCT halfstrip", "CSC LCT strip", 224, -0.5, 223.5);
    lct_bend_data_[iType] = ibooker.book1D("CSC LCT bend", "CSC LCT bend", 3, 0.5, 2.5);
    lct_bx_data_[iType] = ibooker.book1D("CSC LCT bx", "CSC LCT bx", 20, -0.5, 19.5);
  }
}

void L1TdeCSCTPG::analyze(const edm::Event& e, const edm::EventSetup& c) {
  if (verbose_)
    edm::LogInfo("L1TdeCSCTPG") << "L1TdeCSCTPG: analyzing collections" << std::endl;

  // handles
  edm::Handle<CSCALCTDigiCollection> dataALCTs;
  edm::Handle<CSCALCTDigiCollection> emulALCTs;
  edm::Handle<CSCCLCTDigiCollection> dataCLCTs;
  edm::Handle<CSCCLCTDigiCollection> emulCLCTs;
  edm::Handle<CSCCorrelatedLCTDigiCollection> dataLCTs;
  edm::Handle<CSCCorrelatedLCTDigiCollection> emulLCTs;

  e.getByToken(dataALCT_token_, dataALCTs);
  e.getByToken(emulALCT_token_, emulALCTs);
  e.getByToken(dataCLCT_token_, dataCLCTs);
  e.getByToken(emulCLCT_token_, emulCLCTs);
  e.getByToken(dataLCT_token_, dataLCTs);
  e.getByToken(emulLCT_token_, emulLCTs);

  for (auto it = dataALCTs->begin(); it != dataALCTs->end(); it++) {
    auto range = dataALCTs->get((*it).first);
    const int type = ((*it).first).iChamberType() - 1;
    for (auto alct = range.first; alct != range.second; alct++) {
      alct_quality_data_[type]->Fill(alct->getQuality());
      alct_wiregroup_data_[type]->Fill(alct->getKeyWG());
      alct_bx_data_[type]->Fill(alct->getBX());
    }
  }

  for (auto it = emulALCTs->begin(); it != emulALCTs->end(); it++) {
    auto range = emulALCTs->get((*it).first);
    const int type = ((*it).first).iChamberType() - 1;
    for (auto alct = range.first; alct != range.second; alct++) {
      alct_quality_emul_[type]->Fill(alct->getQuality());
      alct_wiregroup_emul_[type]->Fill(alct->getKeyWG());
      alct_bx_emul_[type]->Fill(alct->getBX());
    }
  }

  for (auto it = dataCLCTs->begin(); it != dataCLCTs->end(); it++) {
    auto range = dataCLCTs->get((*it).first);
    const int type = ((*it).first).iChamberType() - 1;
    for (auto clct = range.first; clct != range.second; clct++) {
      clct_pattern_data_[type]->Fill(clct->getPattern());
      clct_quality_data_[type]->Fill(clct->getQuality());
      clct_halfstrip_data_[type]->Fill(clct->getKeyStrip());
      clct_bend_data_[type]->Fill(clct->getBend());
      clct_bx_data_[type]->Fill(clct->getBX());
    }
  }

  for (auto it = emulCLCTs->begin(); it != emulCLCTs->end(); it++) {
    auto range = emulCLCTs->get((*it).first);
    const int type = ((*it).first).iChamberType() - 1;
    for (auto clct = range.first; clct != range.second; clct++) {
      clct_pattern_emul_[type]->Fill(clct->getPattern());
      clct_quality_emul_[type]->Fill(clct->getQuality());
      clct_halfstrip_emul_[type]->Fill(clct->getKeyStrip());
      clct_bend_emul_[type]->Fill(clct->getBend());
      clct_bx_emul_[type]->Fill(clct->getBX());
    }
  }

  for (auto it = dataLCTs->begin(); it != dataLCTs->end(); it++) {
    auto range = dataLCTs->get((*it).first);
    const int type = ((*it).first).iChamberType() - 1;
    for (auto lct = range.first; lct != range.second; lct++) {
      lct_pattern_data_[type]->Fill(lct->getCLCTPattern());
      lct_quality_data_[type]->Fill(lct->getQuality());
      lct_wiregroup_data_[type]->Fill(lct->getKeyWG());
      lct_halfstrip_data_[type]->Fill(lct->getStrip());
      lct_bend_data_[type]->Fill(lct->getBend());
      lct_bx_data_[type]->Fill(lct->getBX());
    }
  }

  for (auto it = emulLCTs->begin(); it != emulLCTs->end(); it++) {
    auto range = emulLCTs->get((*it).first);
    const int type = ((*it).first).iChamberType() - 1;
    for (auto lct = range.first; lct != range.second; lct++) {
      lct_pattern_emul_[type]->Fill(lct->getCLCTPattern());
      lct_quality_emul_[type]->Fill(lct->getQuality());
      lct_wiregroup_emul_[type]->Fill(lct->getKeyWG());
      lct_halfstrip_emul_[type]->Fill(lct->getStrip());
      lct_bend_emul_[type]->Fill(lct->getBend());
      lct_bx_emul_[type]->Fill(lct->getBX());
    }
  }
}
