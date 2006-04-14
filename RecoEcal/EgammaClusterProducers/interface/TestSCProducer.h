#ifndef RecoEcal_EgammaClusterProducers_TestSCProducer_h_
#define RecoEcal_EgammaClusterProducers_TestSCProducer_h_

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


//


class TestSCProducer : public edm::EDProducer 
{
  
  public:

      TestSCProducer(const edm::ParameterSet& ps);

      ~TestSCProducer();

      virtual void produce(edm::Event&, const edm::EventSetup&);

   private:

      int nMaxPrintout_; // max # of printouts
      int nEvt_;         // internal counter of events
 
      std::string superclusterCollection_;

      bool counterExceeded() const { return ((nEvt_ > nMaxPrintout_) || (nMaxPrintout_ < 0));}
};


#endif
