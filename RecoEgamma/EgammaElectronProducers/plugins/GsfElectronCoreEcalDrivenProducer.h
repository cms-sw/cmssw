#ifndef GsfElectronCoreEcalDrivenProducer_h
#define GsfElectronCoreEcalDrivenProducer_h

#include "GsfElectronCoreBaseProducer.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronCoreFwd.h"

class GsfElectronCoreEcalDrivenProducer : public GsfElectronCoreBaseProducer
 {
  public:

    //static void fillDescriptions( edm::ConfigurationDescriptions & ) ;

    explicit GsfElectronCoreEcalDrivenProducer( const edm::ParameterSet & conf ) ;
    virtual ~GsfElectronCoreEcalDrivenProducer() ;
    virtual void produce( edm::Event&, const edm::EventSetup & ) ;

  private:

    void produceEcalDrivenCore( const reco::GsfTrackRef & gsfTrackRef, reco::GsfElectronCoreCollection * electrons ) ;

 } ;

#endif
