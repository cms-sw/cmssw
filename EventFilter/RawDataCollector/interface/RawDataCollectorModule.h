#ifndef RawDataCollectorModule_H
#define RawDataCollectorModule_H


/** \class RawDataCollectorModule
 *
 */

#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/EDProducer.h>

class RawDataCollectorModule: public edm::EDProducer {
public:
    
    ///Constructor
    RawDataCollectorModule(const edm::ParameterSet& pset);
    
    ///Destructor
    virtual ~RawDataCollectorModule();
    
    void produce(edm::Event & e, const edm::EventSetup& c); 
          
private:

    bool useCurrentProcessOnly_ ; 
};

#endif
