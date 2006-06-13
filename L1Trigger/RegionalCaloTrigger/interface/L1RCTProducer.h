#ifndef L1RCTProducer_h
#define L1RCTProducer_h
#include "L1Trigger/RegionalCaloTrigger/interface/L1RCT.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/EDProduct.h"
//#include "DataFormats/RegionalCaloTrigger/interface/L1RCTEcal.h"
//#include "DataFormats/RegionalCaloTrigger/interface/L1RCTHcal.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"

class L1RCTProducer : public edm::EDProducer
{
 public:
  explicit L1RCTProducer(const edm::ParameterSet& ps);
  virtual ~L1RCTProducer();
  virtual void produce(edm::Event& e, const edm::EventSetup& c);
 private:
  L1RCT rct;
  std::string src;
};
#endif
