#ifndef L1RCTInputProducer_h
#define L1RCTInputProducer_h 

#include "FWCore/Framework/interface/EDProducer.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>

class L1RCT;
class L1RCTLookupTables;

class L1RCTInputProducer : public edm::EDProducer
{
 public:
  explicit L1RCTInputProducer(const edm::ParameterSet& ps);
  virtual ~L1RCTInputProducer();
  virtual void produce(edm::Event& e, const edm::EventSetup& c);
 private:
  L1RCTLookupTables* rctLookupTables;
  L1RCT* rct;
  bool useEcal;
  bool useHcal;
  edm::InputTag ecalDigisLabel;
  edm::InputTag hcalDigisLabel;
};
#endif
