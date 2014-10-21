
#ifndef GsfElectronBaseProducer_h
#define GsfElectronBaseProducer_h

#include "RecoEgamma/EgammaElectronAlgos/interface/GsfElectronAlgo.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

#include "FWCore/Framework/interface/stream/EDProducer.h"
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

class GsfElectronBaseProducer : public edm::stream::EDProducer<>
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
    const edm::OrphanHandle<reco::GsfElectronCollection> & orphanHandle() const { return orphanHandle_;}

    // configurables
    GsfElectronAlgo::InputTagsConfiguration inputCfg_ ;
    GsfElectronAlgo::StrategyConfiguration strategyCfg_ ;
    GsfElectronAlgo::CutsConfiguration cutsCfg_ ;
    GsfElectronAlgo::CutsConfiguration cutsCfgPflow_ ;
    ElectronHcalHelper::Configuration hcalCfg_ ;
    ElectronHcalHelper::Configuration hcalCfgPflow_ ;
    SoftElectronMVAEstimator::Configuration mvaCfg_ ;
  private :

    // check expected configuration of previous modules
    bool ecalSeedingParametersChecked_ ;
    void checkEcalSeedingParameters( edm::ParameterSet const & ) ;
    edm::OrphanHandle<reco::GsfElectronCollection> orphanHandle_;

 } ;

#endif
