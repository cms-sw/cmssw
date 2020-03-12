#ifndef ESOccupancyTask_H
#define ESOccupancyTask_H

#include "DQMServices/Core/interface/DQMOneEDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

class ESOccupancyTask : public DQMOneEDAnalyzer<> {
public:
  ESOccupancyTask(const edm::ParameterSet& ps);
  ~ESOccupancyTask() override {}

private:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------
  edm::EDGetTokenT<ESRecHitCollection> rechittoken_;
  std::string prefixME_;

  MonitorElement* hRecOCC_[2][2];
  MonitorElement* hSelOCC_[2][2];
  MonitorElement* hRecNHit_[2][2];
  MonitorElement* hEnDensity_[2][2];
  MonitorElement* hSelEnDensity_[2][2];
  MonitorElement* hGoodRecNHit_[2][2];
  MonitorElement* hSelEng_[2][2];
  MonitorElement* hEng_[2][2];
  MonitorElement* hEvEng_[2][2];
  MonitorElement* hE1E2_[2];

  int runNum_, eCount_;
};

#endif
