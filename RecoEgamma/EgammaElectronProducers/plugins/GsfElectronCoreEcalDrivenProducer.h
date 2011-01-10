#ifndef GsfElectronCoreEcalDrivenProducer_h
#define GsfElectronCoreEcalDrivenProducer_h

#include "GsfElectronCoreBaseProducer.h"

class GsfElectronCoreEcalDrivenProducer : public GsfElectronCoreBaseProducer
 {
  public:

    //static void fillDescriptions( edm::ConfigurationDescriptions & ) ;

    explicit GsfElectronCoreEcalDrivenProducer( const edm::ParameterSet & conf ) ;
    virtual ~GsfElectronCoreEcalDrivenProducer() ;
    virtual void produce( edm::Event&, const edm::EventSetup & ) ;

 } ;

#endif
