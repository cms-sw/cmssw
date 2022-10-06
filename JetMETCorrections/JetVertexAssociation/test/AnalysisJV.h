#ifndef AnalysisJV_H
#define AnalysisJV_H

// system include files
#include <memory>
#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>
#include <vector>
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "TFile.h"
#include "TH1.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"

//
// class decleration
//

class AnalysisJV : public edm::one::EDAnalyzer<> {
public:
  explicit AnalysisJV(const edm::ParameterSet&);
  ~AnalysisJV();

private:
  virtual void beginJob();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob();
  std::string fOutputFileName;

  TFile* fOutputFile;
  TH1D* fHistAlpha;

  typedef std::vector<double> ResultCollection1;
  typedef std::vector<bool> ResultCollection2;

  edm::EDGetTokenT<ResultCollection1> fResult1Token;
  edm::EDGetTokenT<ResultCollection2> fResult2Token;
  edm::EDGetTokenT<reco::CaloJetCollection> fCaloJetsToken;
};
#endif
