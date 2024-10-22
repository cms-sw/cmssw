/*
 * \file L1TCSCTPG.cc
 *
 * \author J. Berryhill
 *
 */

#include "DQM/L1TMonitor/interface/L1TCSCTPG.h"

using namespace std;
using namespace edm;

L1TCSCTPG::L1TCSCTPG(const ParameterSet& ps)
    : csctpgSource_(ps.getParameter<InputTag>("csctpgSource")),
      csctpgSource_token_(consumes<CSCCorrelatedLCTDigiCollection>(ps.getParameter<InputTag>("csctpgSource"))) {
  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  if (verbose_)
    cout << "L1TCSCTPG: constructor...." << endl;

  outputFile_ = ps.getUntrackedParameter<string>("outputFile", "");
  if (!outputFile_.empty()) {
    cout << "L1T Monitoring histograms will be saved to " << outputFile_.c_str() << endl;
  }

  bool disable = ps.getUntrackedParameter<bool>("disableROOToutput", false);
  if (disable) {
    outputFile_ = "";
  }
}

L1TCSCTPG::~L1TCSCTPG() {}

void L1TCSCTPG::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup) {
  nev_ = 0;
  ibooker.setCurrentFolder("L1T/L1TCSCTPG");

  csctpgpattern = ibooker.book1D("CSC TPG hit pattern", "CSC TPG hit pattern", 8, -0.5, 7.5);
  csctpgquality = ibooker.book1D("CSC TPG quality", "CSC TPG quality", 16, 0.5, 16.5);
  csctpgwg = ibooker.book1D("CSC TPG wire group", "CSC TPG wire group", 116, -0.5, 115.5);
  csctpgstrip = ibooker.book1D("CSC TPG strip", "CSC TPG strip", 160, -0.5, 159.5);
  csctpgstriptype = ibooker.book1D("CSC TPG strip type", "CSC TPG strip type", 2, 0.5, 1.5);
  csctpgbend = ibooker.book1D("CSC TPG bend", "CSC TPG bend", 3, 0.5, 2.5);
  csctpgbx = ibooker.book1D("CSC TPG bx", "CSC TPG bx", 20, -0.5, 19.5);
}

void L1TCSCTPG::analyze(const Event& e, const EventSetup& c) {
  nev_++;
  if (verbose_)
    cout << "L1TCSCTPG: analyze...." << endl;

  Handle<CSCCorrelatedLCTDigiCollection> pCSCTPGcorrlcts;
  e.getByToken(csctpgSource_token_, pCSCTPGcorrlcts);

  if (!pCSCTPGcorrlcts.isValid()) {
    edm::LogInfo("DataNotFound") << "can't find CSCCorrelatedLCTDigiCollection with label " << csctpgSource_.label();
    return;
  }

  for (CSCCorrelatedLCTDigiCollection::DigiRangeIterator cscItr1 = pCSCTPGcorrlcts->begin();
       cscItr1 != pCSCTPGcorrlcts->end();
       cscItr1++) {
    CSCCorrelatedLCTDigiCollection::Range range1 = pCSCTPGcorrlcts->get((*cscItr1).first);
    for (CSCCorrelatedLCTDigiCollection::const_iterator lctItr1 = range1.first; lctItr1 != range1.second; lctItr1++) {
      csctpgpattern->Fill(lctItr1->getCLCTPattern());
      if (verbose_) {
        std::cout << "CSC TPG CLCT pattern " << lctItr1->getCLCTPattern() << std::endl;
      }

      csctpgquality->Fill(lctItr1->getQuality());
      if (verbose_) {
        std::cout << "CSC LCT quality " << lctItr1->getQuality() << std::endl;
      }

      csctpgwg->Fill(lctItr1->getKeyWG());
      if (verbose_) {
        std::cout << "CSC LCT wire group " << lctItr1->getKeyWG() << std::endl;
      }

      csctpgstrip->Fill(lctItr1->getStrip());
      if (verbose_) {
        std::cout << "CSC LCT strip " << lctItr1->getStrip() << std::endl;
      }

      csctpgstriptype->Fill(lctItr1->getStripType());
      if (verbose_) {
        std::cout << "CSC LCT strip type" << lctItr1->getStripType() << std::endl;
      }

      csctpgbend->Fill(lctItr1->getBend());
      if (verbose_) {
        std::cout << "CSC LCT bend " << lctItr1->getBend() << std::endl;
      }

      csctpgbx->Fill(lctItr1->getBX());
      if (verbose_) {
        std::cout << "CSC LCT bx " << lctItr1->getBX() << std::endl;
      }
    }
  }
}
