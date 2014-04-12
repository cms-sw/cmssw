#ifndef ESDataCertificationTask_h
#define ESDataCertificationTask_h

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class ESDataCertificationTask: public edm::EDAnalyzer{

 public:

  ESDataCertificationTask(const edm::ParameterSet& ps);
  virtual ~ESDataCertificationTask();

 protected:

  void analyze(const edm::Event& e, const edm::EventSetup& c);
  void beginJob(void);
  void endJob(void);
  void beginLuminosityBlock(const edm::LuminosityBlock& lumiBlock, const  edm::EventSetup& iSetup);
  void endLuminosityBlock(const edm::LuminosityBlock&  lumiBlock, const  edm::EventSetup& iSetup);
  void reset(void);
  void cleanup(void);
  
 private:
  
  DQMStore* dqmStore_;
  
  std::string prefixME_;
  
  bool enableCleanup_;
  
  bool mergeRuns_;
  
  MonitorElement* meESDataCertificationSummary_;
  MonitorElement* meESDataCertificationSummaryMap_;
  
};

#endif
