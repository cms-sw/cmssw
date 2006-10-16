#ifndef L1RCTTestProducer_h
#define L1RCTTestProducer_h

#include "L1Trigger/RegionalCaloTrigger/interface/L1RCT.h"

#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/EDProduct.h"

//#include "DataFormats/RegionalCaloTrigger/interface/L1RCTEcal.h"
//#include "DataFormats/RegionalCaloTrigger/interface/L1RCTHcal.h"

//#include "DataFormats/L1CaloTrigger/interface/L1CaloEmCand.h"
//#include "DataFormats/L1CaloTrigger/interface/L1CaloRegion.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

class L1RCTTestProducer : public edm::EDProducer
{
 public:
  explicit L1RCTTestProducer(const edm::ParameterSet& ps);
  virtual ~L1RCTTestProducer();
  virtual void produce(edm::Event& e, const edm::EventSetup& c);
 private:
  L1RCT* rct;
  std::string src;
  bool orcaFileInput;
};
#endif
