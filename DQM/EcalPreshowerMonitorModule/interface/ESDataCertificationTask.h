#ifndef ESDataCertificationTask_h
#define ESDataCertificationTask_h

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class DQMStore;
class MonitorElement;

class ESDataCertificationTask: public edm::EDAnalyzer{

 public:

  ESDataCertificationTask(const edm::ParameterSet& ps);
  virtual ~ESDataCertificationTask();

 protected:

  void analyze(const edm::Event& e, const edm::EventSetup& c);
  void beginJob();
  void endJob();
  void beginLuminosityBlock(const edm::LuminosityBlock& lumiBlock, const  edm::EventSetup& iSetup);
  void endLuminosityBlock(const edm::LuminosityBlock&  lumiBlock, const  edm::EventSetup& iSetup);
  void reset();
  void cleanup();
  
 private:
  
  DQMStore* dqmStore_;
  
  std::string prefixME_;
  
  bool enableCleanup_;
  
  bool mergeRuns_;
  
  MonitorElement* meESDataCertificationSummary_;
  MonitorElement* meESDataCertificationSummaryMap_;
  
};

#endif
