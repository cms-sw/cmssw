#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelPerformanceSummary.h"


class SiPixelHistoricInfoDQMClient : public edm::EDAnalyzer {
  typedef std::vector<std::string> vstring; 

public:
  explicit SiPixelHistoricInfoDQMClient(const edm::ParameterSet&);
 ~SiPixelHistoricInfoDQMClient();

private:
  virtual void beginJob(const edm::EventSetup&);
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob();

  void retrieveMEs();
  void fillPerformanceSummary() const;

  void writeDB(std::string) const; 
  void saveFile(std::string) const; 

private: 
  bool printDebug_;
  bool writeHisto_;
  std::string outputDir_; 

  edm::ParameterSet parameterSet_;
  DQMStore* dbe_;

  int nEventsInRun; 
  std::map< uint32_t, std::vector<MonitorElement*> > ClientPointersToModuleMEs;
  SiPixelPerformanceSummary* performanceSummary;
};

