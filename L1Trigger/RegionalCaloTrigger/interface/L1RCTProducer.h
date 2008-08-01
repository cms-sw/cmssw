#ifndef L1RCTProducer_h
#define L1RCTProducer_h 

#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/EDProduct.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>

class L1RCT;
class L1RCTLookupTables;

class L1RCTProducer : public edm::EDProducer
{
 public:
  explicit L1RCTProducer(const edm::ParameterSet& ps);
  virtual ~L1RCTProducer();
  virtual void beginJob(const edm::EventSetup& c);
  virtual void produce(edm::Event& e, const edm::EventSetup& c);
 private:
  L1RCTLookupTables* rctLookupTables;
  L1RCT* rct;
  bool useEcal;
  bool useHcal;
  edm::InputTag ecalDigisLabel;
  edm::InputTag hcalDigisLabel;
  std::string ecalESLabel;
  std::string hcalESLabel;
  bool useHcalCosmicTiming;
  bool useEcalCosmicTiming;
  unsigned preSamples;
  unsigned postSamples;
};
#endif
