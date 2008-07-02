#ifndef _SeedProducer_h_
#define _SeedProducer_h_

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>
#include <vector>

class SeedProducer : public edm::EDProducer {
 public:
   explicit SeedProducer(const edm::ParameterSet& ps_);
   ~SeedProducer();
   virtual void produce(edm::Event& ev, const edm::EventSetup& es);

 private:
   std::vector<std::string> tripletList;

   const edm::ParameterSet ps;
};
#endif

