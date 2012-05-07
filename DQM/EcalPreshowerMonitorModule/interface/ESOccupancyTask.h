#ifndef ESOccupancyTask_H
#define ESOccupancyTask_H

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class MonitorElement;
class DQMStore;

class ESOccupancyTask : public edm::EDAnalyzer {

 public:
  
  ESOccupancyTask(const edm::ParameterSet& ps);
  virtual ~ESOccupancyTask();
  
 private:
  
  virtual void beginJob(void);
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob(void) ;
  
  // ----------member data ---------------------------
  edm::InputTag rechitlabel_;
  edm::InputTag digilabel_;
  std::string prefixME_;
  
  DQMStore* dqmStore_;
  MonitorElement* hRecOCC_[2][2];
  MonitorElement* hSelOCC_[2][2];
  MonitorElement* hRecNHit_[2][2];
  MonitorElement* hEnDensity_[2][2];
  MonitorElement* hSelEnDensity_[2][2];
  MonitorElement* hDigiNHit_[2][2];
  MonitorElement* hSelEng_[2][2];
  MonitorElement* hEng_[2][2];
  MonitorElement* hEvEng_[2][2];
  MonitorElement* hE1E2_[2];

  int runNum_, eCount_; 
  
};

#endif
