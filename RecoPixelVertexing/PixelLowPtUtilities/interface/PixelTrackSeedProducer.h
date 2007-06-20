#ifndef _PixelTrackSeedProducer_h_
#define _PixelTrackSeedProducer_h_

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>
#include <vector>
using namespace std;

class PixelTrackSeedProducer : public edm::EDProducer {
 public:
   explicit PixelTrackSeedProducer(const edm::ParameterSet& ps);
   ~PixelTrackSeedProducer();
   virtual void produce(edm::Event& ev, const edm::EventSetup& es);

 private:
   vector<string> tripletList;
};
#endif

