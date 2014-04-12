// -*- C++ -*-
//
// Package:    EventCountProducer
// Class:      EventCountProducer
// 
/**\class EventCountProducer EventCountProducer.cc CommonTools/UtilAlgos/plugins/EventCountProducer.cc

Description: An event counter that can store the number of events in the lumi block 

*/


// system include files
#include <memory>
#include <vector>
#include <algorithm>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/MergeableCounter.h"


class EventCountProducer : public edm::one::EDProducer<edm::one::WatchLuminosityBlocks,
                                                       edm::EndLuminosityBlockProducer> {
public:
  explicit EventCountProducer(const edm::ParameterSet&);
  ~EventCountProducer();

private:
  virtual void produce(edm::Event &, const edm::EventSetup&) override;
  virtual void beginLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup&) override;
  virtual void endLuminosityBlock(edm::LuminosityBlock const&, const edm::EventSetup&) override;
  virtual void endLuminosityBlockProduce(edm::LuminosityBlock &, const edm::EventSetup&) override;
      
  // ----------member data ---------------------------

  unsigned int eventsProcessedInLumi_;

};



using namespace edm;
using namespace std;



EventCountProducer::EventCountProducer(const edm::ParameterSet& iConfig){
  produces<edm::MergeableCounter, edm::InLumi>();
}


EventCountProducer::~EventCountProducer(){}


void
EventCountProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup){
  eventsProcessedInLumi_++;
  return;
}


void 
EventCountProducer::beginLuminosityBlock(const LuminosityBlock & theLuminosityBlock, const EventSetup & theSetup) {
  eventsProcessedInLumi_ = 0;
  return;
}

void 
EventCountProducer::endLuminosityBlock(LuminosityBlock const& theLuminosityBlock, const EventSetup & theSetup) {
}

void 
EventCountProducer::endLuminosityBlockProduce(LuminosityBlock & theLuminosityBlock, const EventSetup & theSetup) {
  LogTrace("EventCounting") << "endLumi: adding " << eventsProcessedInLumi_ << " events" << endl;

  auto_ptr<edm::MergeableCounter> numEventsPtr(new edm::MergeableCounter);
  numEventsPtr->value = eventsProcessedInLumi_;
  theLuminosityBlock.put(numEventsPtr);

  return;
}



//define this as a plug-in
DEFINE_FWK_MODULE(EventCountProducer);
