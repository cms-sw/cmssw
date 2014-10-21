#ifndef ESTrendTask_H
#define ESTrendTask_H

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "TProfile.h"

class MonitorElement;

class ESTrendTask: public DQMEDAnalyzer{

 public:

  // Constructor
  ESTrendTask(const edm::ParameterSet& ps);

  // Destructor
  virtual ~ESTrendTask() {}

 protected:

  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

  // Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  // EndJob
  void endJob(void);

  // BeginRun
  void dqmBeginRun(const edm::Run & r, const edm::EventSetup & c) override;

  // Update time check
  void updateTime(const edm::Event&);

  // Shift bins of TProfile to the right
  void shift2Right(TProfile* p, int bins=1);

  // Shift bins of TProfile to the left
  void shift2Left(TProfile* p, int bins=1);

 private:

  int ievt_;

  std::string prefixME_;

  edm::EDGetTokenT<ESRecHitCollection> rechittoken_;
  edm::EDGetTokenT<ESRawDataCollection> dccCollections_;

  MonitorElement* hESRecHitTrend_[2][2];
  MonitorElement* hESSLinkErrTrend_;
  MonitorElement* hESFiberErrTrend_;

  MonitorElement* hESRecHitTrendHr_[2][2];
  MonitorElement* hESSLinkErrTrendHr_;
  MonitorElement* hESFiberErrTrendHr_;

  long int start_time_;
  long int current_time_;
  long int last_time_;

};

#endif
