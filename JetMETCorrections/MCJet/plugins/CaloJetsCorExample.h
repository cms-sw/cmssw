#ifndef CALOJETSCOREXAMPLE_H
#define CALOJETSCOREXAMPLE_H

#include "TH1.h"
#include "TFile.h"
#include "TNamed.h"
#include <map>
#include "FWCore/Framework/interface/EDAnalyzer.h"

namespace cms
{
class CaloJetsCorExample : public edm::EDAnalyzer 
{
  public:
    explicit CaloJetsCorExample(edm::ParameterSet const& cfg);
    virtual void analyze(edm::Event const& e, edm::EventSetup const& iSetup);
    virtual void endJob();
    CaloJetsCorExample();
    void fillHist(const TString& histName, const Double_t& value); 
      
private:
  std::string GenJetAlgorithm_;
  std::string CaloJetAlgorithm_;
  std::string CorJetAlgorithm_;  
  std::string JetCorrectionService_;
  std::string HistogramFile_; 
  std::map<TString, TH1*> m_HistNames;
  TFile* m_file;
};
}
#endif
