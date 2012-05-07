#ifndef ESTrendTask_H
#define ESTrendTask_H

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TProfile.h"

class MonitorElement;
class DQMStore;

class ESTrendTask: public edm::EDAnalyzer{

 public:

  // Constructor
  ESTrendTask(const edm::ParameterSet& ps);

  // Destructor
  virtual ~ESTrendTask();

 protected:

  // Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);

  // BeginJob
  void beginJob(void);

  // EndJob
  void endJob(void);

  // BeginRun
  void beginRun(const edm::Run & r, const edm::EventSetup & c);

  // EndRun
  void endRun(const edm::Run & r, const edm::EventSetup & c);

  // Reset
  void reset(void);

  // Setup
  void setup(void);

  // Cleanup
  void cleanup(void);

  // Update time check
  void updateTime(void);

  // Shift bins of TProfile to the right
  void shift2Right(TProfile* p, int bins=1);

  // Shift bins of TProfile to the left
  void shift2Left(TProfile* p, int bins=1);

 private:

  int ievt_;

  DQMStore* dqmStore_;

  std::string prefixME_;

  bool enableCleanup_;

  bool mergeRuns_;

  edm::InputTag rechitlabel_;
  edm::InputTag dccCollections_;

  MonitorElement* hESRecHitTrend_[2][2];
  MonitorElement* hESSLinkErrTrend_;
  MonitorElement* hESFiberErrTrend_;

  MonitorElement* hESRecHitTrendHr_[2][2];
  MonitorElement* hESSLinkErrTrendHr_;
  MonitorElement* hESFiberErrTrendHr_;

  bool init_;

  long int start_time_;
  long int current_time_;
  long int last_time_;

};

#endif
