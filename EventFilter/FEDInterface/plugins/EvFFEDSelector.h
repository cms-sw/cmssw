#ifndef EVENTFILTER_FEDINTERFACE_PLUGINS_EVFFEDSELECTOR
#define EVENTFILTER_FEDINTERFACE_PLUGINS_EVFFEDSELECTOR

#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/EDProducer.h>
#include <FWCore/Framework/interface/Event.h>

#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include <vector>

namespace evf{
  class EvFFEDSelector : public edm::EDProducer
    {
    public:
      
      explicit EvFFEDSelector( const edm::ParameterSet& );
      ~EvFFEDSelector(){};
      
      void produce(edm::Event & e, const edm::EventSetup& c);
      
    private:
      edm::InputTag label_;
      edm::EDGetTokenT<FEDRawDataCollection> token_;
      std::vector<unsigned int> fedlist_;
      
    };
}

#endif
