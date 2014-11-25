
#ifndef GsfElectronBaseProducer_h
#define GsfElectronBaseProducer_h

#include "RecoEgamma/EgammaElectronAlgos/interface/GsfElectronAlgo.h"

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

namespace reco
 {
  class GsfElectron ;
 }

namespace edm
 {
  class ParameterSet ;
  class ConfigurationDescriptions ;
 }

#include "RecoEgamma/EgammaElectronAlgos/interface/GsfElectronAlgo.h"
#include "DataFormats/Common/interface/Handle.h"

class GsfElectronBaseProducer : public edm::EDProducer
 {
  public:

    static void fillDescription( edm::ParameterSetDescription & ) ;

    explicit GsfElectronBaseProducer( const edm::ParameterSet & ) ;
    virtual ~GsfElectronBaseProducer() ;


  protected:

    GsfElectronAlgo * algo_ ;

    void beginEvent( edm::Event &, const edm::EventSetup & ) ;
    void fillEvent( edm::Event & ) ;
    void endEvent() ;
    reco::GsfElectron * newElectron() { return 0 ; }

    // configurables
    GsfElectronAlgo::InputTagsConfiguration inputCfg_ ;
    GsfElectronAlgo::StrategyConfiguration strategyCfg_ ;
    GsfElectronAlgo::CutsConfiguration cutsCfg_ ;
    GsfElectronAlgo::CutsConfiguration cutsCfgPflow_ ;
    //ElectronHcalHelper::Configuration hcalCfg_ ;
    ElectronHcalHelper::Configuration hcalCfgBarrel_ ;
    ElectronHcalHelper::Configuration hcalCfgEndcap_ ;
    ElectronHcalHelper::Configuration hcalCfgPflow_ ;

  private :

    // check expected configuration of previous modules
    bool ecalSeedingParametersChecked_ ;
    void checkEcalSeedingParameters( edm::ParameterSetID const & ) ;

 } ;

#endif
