#ifndef ESOccupancyTask_H
#define ESOccupancyTask_H

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

class MonitorElement;

class ESOccupancyTask : public DQMEDAnalyzer {

 public:
  
  ESOccupancyTask(const edm::ParameterSet& ps);
  virtual ~ESOccupancyTask() {}
  
 private:

  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  
  // ----------member data ---------------------------
  edm::EDGetTokenT<ESRecHitCollection> rechittoken_;
  std::string prefixME_;
  
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
