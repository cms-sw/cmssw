#ifndef EBTrendClient_H
#define EBTrendClient_H

/*
 * \file EBTrendClient.h
 *
 * $Date: 2010/03/28 09:05:23 $
 * $Revision: 1.3 $
 * \author Dongwook Jang, Soon Yung Jun
 *
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TProfile.h"
#include "TH1F.h"
#include "TObject.h"

class MonitorElement;
class DQMStore;

const int nHists_ = 6;

class EBTrendClient: public edm::EDAnalyzer{

 public:

  // Constructor
  EBTrendClient(const edm::ParameterSet& ps);

  // Destructor
  virtual ~EBTrendClient();

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


 private:

  int ievt_;

  DQMStore* dqmStore_;

  std::string prefixME_;

  bool enableCleanup_;

  bool mergeRuns_;

  bool verbose_;

  MonitorElement* meanMinutely_[nHists_];
  MonitorElement* sigmaMinutely_[nHists_];

  MonitorElement* meanHourly_[nHists_];
  MonitorElement* sigmaHourly_[nHists_];

  bool init_;

  int start_time_;
  int current_time_;
  int last_time_;

  std::string moduleNames_[nHists_];
  std::string histTitles_[nHists_];

  double mean_[nHists_];
  double rms_[nHists_];

  TObject* previousHist_[nHists_];
  TObject* currentHist_[nHists_];

};

#endif
