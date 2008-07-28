#ifndef PFJETSCOREXAMPLE_H
#define PFJETSCOREXAMPLE_H

#include "TH1.h"
#include "TFile.h"
#include "TNamed.h"
#include <map>
#include "FWCore/Framework/interface/EDAnalyzer.h"

namespace cms
{
class PFJetsCorExample : public edm::EDAnalyzer 
{
  public:
    explicit PFJetsCorExample(edm::ParameterSet const& cfg);
    virtual void analyze(edm::Event const& e, edm::EventSetup const& iSetup);
    virtual void endJob();
    PFJetsCorExample();
    void fillHist(const TString& histName, const Double_t& value); 
      
private:
  std::string GenJetAlgorithm_;
  std::string PFJetAlgorithm_; 
  std::string JetCorrectionService_;
  std::string HistogramFile_; 
  std::map<TString, TH1*> m_HistNames;
  TFile* m_file;
};
}
#endif
