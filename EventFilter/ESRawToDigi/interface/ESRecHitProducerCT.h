#ifndef ESRecHitProducerCT_H
#define ESRecHitProducerCT_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "EventFilter/ESRawToDigi/interface/ESRecHitSimAlgoCT.h"

#include "TH1F.h"
#include "TFile.h"

using namespace std;
using namespace edm;

class ESRecHitProducerCT : public edm::EDProducer {

 public:

  explicit ESRecHitProducerCT(const edm::ParameterSet& ps);
  virtual ~ESRecHitProducerCT();
  virtual void produce(edm::Event& e, const edm::EventSetup& es);
  void DoCommonMode(double det_data[], double *cm1, double *cm2);

 private:

  int gain_;

  InputTag digiCollection_;
  string rechitCollection_;
  
  string pedestalFile_;
  int detType_; 
  int doCM_;  

  TH1F* hist_[2][6][4][5];
  TFile *ped_;

  ESRecHitSimAlgoCT *algo_;

};
#endif
