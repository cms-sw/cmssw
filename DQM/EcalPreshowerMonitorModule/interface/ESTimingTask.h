#ifndef ESTimingTask_H
#define ESTimingTask_H

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CondFormats/ESObjects/interface/ESGain.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "TF1.h"
#include "TH1F.h"

class MonitorElement;

double fitf(double *x, double *par);

class ESTimingTask : public DQMEDAnalyzer {

 public:
  
  ESTimingTask(const edm::ParameterSet& ps);
  virtual ~ESTimingTask();
  
 private:

  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  void set(const edm::EventSetup& es);

  // ----------member data ---------------------------
  edm::EDGetTokenT<ESDigiCollection> digilabel_;
  std::string prefixME_;
  
  MonitorElement* hTiming_[2][2];
  MonitorElement* h2DTiming_;

  edm::ESHandle<ESGain> esgain_;

  TF1 *fit_;
  TH1F *htESP_;
  TH1F *htESM_;

  int runNum_, eCount_; 
  Double_t wc_, n_;
  
};

#endif
