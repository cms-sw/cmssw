#ifndef ESRecHitProducerTB_H
#define ESRecHitProducerTB_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "EventFilter/ESRawToDigi/interface/ESRecHitSimAlgoTB.h"

#include "TH1F.h"
#include "TFile.h"

using namespace std;
using namespace edm;

class ESRecHitProducerTB : public edm::EDProducer {

 public:

  explicit ESRecHitProducerTB(const edm::ParameterSet& ps);
  virtual ~ESRecHitProducerTB();
  virtual void produce(edm::Event& e, const edm::EventSetup& es);
  void DoCommonMode(double det_data[], double *cm);

 private:

  int gain_;

  InputTag digiCollection_;
  string rechitCollection_;
  
  string pedestalFile_;
  int detType_; 
  int doCM_;  
  double sigma_;

  TH1F* hist_[2][4][4];
  TFile *ped_;

  ESRecHitSimAlgoTB *algo_;

};
#endif
