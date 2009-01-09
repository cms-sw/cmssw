#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <TFile.h>
#include <string>


class LikelihoodPdfDBWriter : public edm::EDAnalyzer {
public:
  explicit LikelihoodPdfDBWriter(const edm::ParameterSet&);
  ~LikelihoodPdfDBWriter();
  
  
private:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  /// File with the histograms for the likelihood
  std::string inputFile_;
};

