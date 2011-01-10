
#ifndef GsfElectronProducer_h
#define GsfElectronProducer_h

#include "GsfElectronBaseProducer.h"

class GsfElectronProducer : public GsfElectronBaseProducer
 {
  public:

    //static void fillDescriptions( edm::ConfigurationDescriptions & ) ;

    explicit GsfElectronProducer( const edm::ParameterSet & ) ;
    virtual ~GsfElectronProducer();
    virtual void produce( edm::Event &, const edm::EventSetup & ) ;

 } ;

#endif
