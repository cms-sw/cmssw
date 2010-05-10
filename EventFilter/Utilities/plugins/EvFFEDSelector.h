#ifndef EVENTFILTER_UTILTIES_PLUGINS_EVFFEDSELECTOR
#define EVENTFILTER_UTILTIES_PLUGINS_EVFFEDSELECTOR

#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/EDProducer.h>
#include <FWCore/Framework/interface/Event.h>

#include <vector>

namespace evf{
  class EvFFEDSelector : public edm::EDProducer
    {
    public:
      
      explicit EvFFEDSelector( const edm::ParameterSet& );
      ~EvFFEDSelector(){};
      
      void produce(edm::Event & e, const edm::EventSetup& c);
      
    private:
      std::vector<unsigned int> fedlist_;
      
    };
}

#endif
