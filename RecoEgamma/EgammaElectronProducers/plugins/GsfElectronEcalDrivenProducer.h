
#ifndef GsfElectronEcalDrivenProducer_h
#define GsfElectronEcalDrivenProducer_h

#include "GsfElectronBaseProducer.h"

class GsfElectronEcalDrivenProducer : public GsfElectronBaseProducer
 {
  public:

    //static void fillDescriptions( edm::ConfigurationDescriptions & ) ;

    explicit GsfElectronEcalDrivenProducer( const edm::ParameterSet & ) ;
    virtual ~GsfElectronEcalDrivenProducer() ;
    virtual void produce( edm::Event &, const edm::EventSetup & ) ;

 } ;

#endif
