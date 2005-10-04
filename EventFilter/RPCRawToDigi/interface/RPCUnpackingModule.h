#ifndef EVENTFILTER_RPCUnpackingModule_H
#define EVENTFILTER_RPCUnpackingModule_H


#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/EDProducer.h>

#include <iostream>


  class RPCUnpackingModule: public edm::EDProducer {
  public:
    /// Constructor
    RPCUnpackingModule(const edm::ParameterSet& pset);

    /// Destructor
    virtual ~RPCUnpackingModule();
    
    /// Produce digis out of raw data
    void produce(edm::Event & e, const edm::EventSetup& c);

  };

DEFINE_FWK_MODULE(RPCUnpackingModule)

#endif
