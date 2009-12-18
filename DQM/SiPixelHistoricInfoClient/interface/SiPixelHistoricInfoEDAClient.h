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

// #include "DQM/SiPixelHistoricInfoClient/interface/SiPixelHistoricInfoWebInterface.h"


class SiPixelHistoricInfoEDAClient : public edm::EDAnalyzer {
public:
  explicit SiPixelHistoricInfoEDAClient(const edm::ParameterSet&);
 ~SiPixelHistoricInfoEDAClient();

private:
  virtual void beginJob();
  virtual void beginRun(const edm::Run&, const edm::EventSetup&);
  virtual void beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&);
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&);
  virtual void endRun(const edm::Run&, const edm::EventSetup&);
  virtual void endJob();

  void retrieveMEs();
  void fillPerformanceSummary() const;
  void writeDB() const; 
  void saveFile(std::string filename) const { dbe_->save(filename); }

private: 
  bool printDebug_;
  bool writeHisto_;
  std::string outputDir_; 

  edm::ParameterSet parameterSet_;
  DQMStore* dbe_;

  bool firstEventInRun; 
  int nEventsInRun; 

  SiPixelHistogramId histogramManager;
  std::map< uint32_t, std::vector<MonitorElement*> > mapOfdetIDtoMEs;
  SiPixelPerformanceSummary* performanceSummary;

  // SiPixelHistoricInfoWebInterface* webInterface_;
  // bool defaultWebPageCreated_; 
};

