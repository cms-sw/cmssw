#ifndef ESFEDIntegrityTask_H
#define ESFEDIntegrityTask_H

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class MonitorElement;
class DQMStore;

class ESFEDIntegrityTask : public edm::EDAnalyzer {
  
 public:
  
  ESFEDIntegrityTask(const edm::ParameterSet& ps);
  virtual ~ESFEDIntegrityTask();
  
 protected:
  
  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);
  
  /// BeginJob
  void beginJob(void);
  
  /// EndJob
  void endJob(void);
  
  /// BeginRun
  void beginRun(const edm::Run & r, const edm::EventSetup & c);
  
  /// EndRun
  void endRun(const edm::Run & r, const edm::EventSetup & c);
  
  /// Reset
  void reset(void);
  
  /// Setup
  void setup(void);
  
  /// Cleanup
  void cleanup(void);
  
 private:
  
  int ievt_;
  
  DQMStore* dqmStore_;
  
  std::string prefixME_;
  std::string fedDirName_;
  bool enableCleanup_;
  bool mergeRuns_;
  bool debug_;

  edm::InputTag dccCollections_;
  edm::InputTag kchipCollections_;
  edm::InputTag FEDRawDataCollection_; 
  
  MonitorElement* meESFedsEntries_;
  MonitorElement* meESFedsFatal_;
  MonitorElement* meESFedsNonFatal_;
  
  bool init_;

};

#endif
