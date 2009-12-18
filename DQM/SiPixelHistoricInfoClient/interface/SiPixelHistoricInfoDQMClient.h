#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQM/SiPixelCommon/interface/SiPixelHistogramId.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelPerformanceSummary.h"


class SiPixelHistoricInfoDQMClient : public edm::EDAnalyzer {
  typedef std::vector<std::string> vstring; 

public:
  explicit SiPixelHistoricInfoDQMClient(const edm::ParameterSet&);
 ~SiPixelHistoricInfoDQMClient() {};

private:
  void beginJob();
  void beginRun(const edm::Run&, const edm::EventSetup&) {};
  void analyze(const edm::Event&, const edm::EventSetup&) {};
  void endRun(const edm::Run&, const edm::EventSetup&);
  void endJob();

  uint32_t getSummaryRegionID(std::string) const; 
  void retrieveMEs();
  void fillPerformanceSummary() const;
  void getSummaryMEmeanRMSnBins(std::vector<MonitorElement*>::const_iterator, 
                                float&, float&, float&) const; 
  void writeDB() const; 
  void saveFile(std::string filename) const { dbe_->save(filename); };

private: 
  bool useSummary_; 
  bool printDebug_;
  bool writeHisto_;
  std::vector<std::string> inputFiles_;
  std::string outputDir_; 

  edm::ParameterSet parameterSet_;
  DQMStore* dbe_;

  SiPixelHistogramId histogramManager;
  std::map< uint32_t, std::vector<MonitorElement*> > mapOfdetIDtoMEs;
  SiPixelPerformanceSummary* performanceSummary;
};

