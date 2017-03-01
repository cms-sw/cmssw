
#ifndef GsfElectronProducer_h
#define GsfElectronProducer_h

#include "GsfElectronBaseProducer.h"

class GsfElectronProducer : public GsfElectronBaseProducer
 {
  public:

    //static void fillDescriptions( edm::ConfigurationDescriptions & ) ;

    explicit GsfElectronProducer( const edm::ParameterSet &, const gsfAlgoHelpers::HeavyObjectCache* ) ;
    virtual ~GsfElectronProducer();
    virtual void produce( edm::Event &, const edm::EventSetup & ) ;

  protected:

    void beginEvent( edm::Event &, const edm::EventSetup & ) ;

  private :

    // check expected configuration of previous modules
    bool pfTranslatorParametersChecked_ ;
    void checkPfTranslatorParameters( edm::ParameterSet const & ) ;
 } ;

#endif
