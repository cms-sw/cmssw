#ifndef GsfElectronCoreProducer_h
#define GsfElectronCoreProducer_h

#include "GsfElectronCoreBaseProducer.h"

class GsfElectronCoreProducer : public GsfElectronCoreBaseProducer
 {
  public:

    //static void fillDescriptions( edm::ConfigurationDescriptions & ) ;

    explicit GsfElectronCoreProducer( const edm::ParameterSet & ) ;
    virtual ~GsfElectronCoreProducer() ;
    virtual void produce( edm::Event &, const edm::EventSetup & ) ;

  private:

    edm::InputTag edCoresTag_ ;
    edm::InputTag pfSuperClustersTag_ ;
    edm::InputTag pfSuperClusterTrackMapTag_ ;

 } ;

#endif
