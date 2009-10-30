#ifndef ESDcsInfoTask_h
#define ESDcsInfoTask_h

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class ESDcsInfoTask: public edm::EDAnalyzer{

 public:
  
  ESDcsInfoTask(const edm::ParameterSet& ps);
  virtual ~ESDcsInfoTask();
  
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
  
  MonitorElement* meESDcsFraction_;
  MonitorElement* meESDcsActiveMap_;
  
};

#endif
