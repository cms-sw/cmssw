
#ifndef GsfElectronEcalDrivenProducer_h
#define GsfElectronEcalDrivenProducer_h

#include "GsfElectronBaseProducer.h"

class GsfElectronEcalDrivenProducer : public GsfElectronBaseProducer
 {
  public:

    explicit GsfElectronEcalDrivenProducer( const edm::ParameterSet &, const gsfAlgoHelpers::HeavyObjectCache* ) ;
    ~GsfElectronEcalDrivenProducer() override ;
    void produce( edm::Event &, const edm::EventSetup & ) override ;

 } ;

#endif
