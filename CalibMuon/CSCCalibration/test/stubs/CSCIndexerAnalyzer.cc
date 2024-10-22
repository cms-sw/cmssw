#include <memory>

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CalibMuon/CSCCalibration/interface/CSCIndexerBase.h"
#include "CalibMuon/CSCCalibration/interface/CSCIndexerRecord.h"
#include <iostream>

class CSCIndexerAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit CSCIndexerAnalyzer(const edm::ParameterSet &);
  ~CSCIndexerAnalyzer() override = default;

private:
  virtual void analyze(const edm::Event &, const edm::EventSetup &) override;
  const edm::ESGetToken<CSCIndexerBase, CSCIndexerRecord> theCSCIndexerToken_;

  std::string algoName;
};

CSCIndexerAnalyzer::CSCIndexerAnalyzer(const edm::ParameterSet &pset) : theCSCIndexerToken_(esConsumes()) {}

void CSCIndexerAnalyzer::analyze(const edm::Event &ev, const edm::EventSetup &iSetup) {
  const int evalues[10] = {1, 2, 1, 2, 1, 2, 1, 2, 1, 2};    // endcap 1=+z, 2=-z
  const int svalues[10] = {1, 1, 1, 1, 4, 4, 4, 4, 4, 4};    // station 1-4
  const int rvalues[10] = {1, 1, 4, 4, 2, 2, 2, 2, 2, 2};    // ring 1-4
  const int cvalues[10] = {1, 1, 1, 1, 1, 1, 36, 36, 1, 1};  // chamber 1-18/36
  const int lvalues[10] = {1, 1, 1, 1, 1, 1, 1, 1, 6, 6};    // layer 1-6
  const int tvalues[10] = {1, 1, 1, 1, 1, 1, 1, 1, 80, 80};  // strip 1-80 (16, 48 64)

  const auto indexer_ = &iSetup.getData(theCSCIndexerToken_);
  algoName = indexer_->name();

  std::cout << "CSCIndexerAnalyzer: analyze sees algorithm " << algoName << " in Event Setup" << std::endl;

  for (int i = 0; i < 10; ++i) {
    int ie = evalues[i];
    int is = svalues[i];
    int ir = rvalues[i];
    int ic = cvalues[i];
    int il = lvalues[i];
    int istrip = tvalues[i];

    std::cout << "CSCIndexerAnalyzer: calling " << algoName << "::stripChannelIndex(" << ie << "," << is << "," << ir
              << "," << ic << "," << il << "," << istrip
              << ") = " << indexer_->stripChannelIndex(ie, is, ir, ic, il, istrip) << std::endl;
  }
}

// define this as a plug-in
DEFINE_FWK_MODULE(CSCIndexerAnalyzer);
