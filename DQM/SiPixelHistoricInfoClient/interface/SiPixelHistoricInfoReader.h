#ifndef SiPixelHistoricInfoReader_H
#define SiPixelHistoricInfoReader_H


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TFile.h"
#include "TObjArray.h"
#include "TH2F.h"


namespace cms {
  class SiPixelHistoricInfoReader : public edm::EDAnalyzer {
  public:
    explicit SiPixelHistoricInfoReader(const edm::ParameterSet&);
            ~SiPixelHistoricInfoReader();

    virtual void beginJob(const edm::EventSetup&);
    virtual void beginRun(const edm::Run&, const edm::EventSetup&) ;
    void analyze(const edm::Event&, const edm::EventSetup&);
    virtual void endRun(const edm::Run&, const edm::EventSetup&) ;
    virtual void endJob();
    
    void fillHistograms(edm::ESHandle<SiPixelPerformanceSummary>);

  private:
    bool printDebug_;
    std::string outputDir_;
    unsigned int presentRun_;
    std::vector<uint32_t> allDetIds;
    TH1F *NumberOfDigisAllDets;
    TObjArray *AllDetHistograms;
    TFile *outputFile;
  };
}


#endif
