
#ifndef GsfElectronBaseProducer_h
#define GsfElectronBaseProducer_h

#include "RecoEgamma/EgammaElectronAlgos/interface/GsfElectronAlgo.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/GsfElectronAlgoHeavyObjectCache.h"

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

class GsfElectronBaseProducer : public edm::stream::EDProducer< edm::GlobalCache<gsfAlgoHelpers::HeavyObjectCache> >
 {
  public:

    static void fillDescriptions( edm::ConfigurationDescriptions & ) ;

    explicit GsfElectronBaseProducer( const edm::ParameterSet &, const gsfAlgoHelpers::HeavyObjectCache* ) ;
    ~GsfElectronBaseProducer() override ;

    static std::unique_ptr<gsfAlgoHelpers::HeavyObjectCache> 
    initializeGlobalCache( const edm::ParameterSet& conf ) {
       return std::make_unique<gsfAlgoHelpers::HeavyObjectCache>(conf);
   }
  
  static void globalEndJob(gsfAlgoHelpers::HeavyObjectCache const* ) {
  }

  protected:

    std::unique_ptr<GsfElectronAlgo> algo_ ;

    void beginEvent( edm::Event &, const edm::EventSetup & ) ;
    void fillEvent( reco::GsfElectronCollection & electrons, edm::Event & event ) ;
    const edm::OrphanHandle<reco::GsfElectronCollection> & orphanHandle() const { return orphanHandle_;}

    // configurables
    GsfElectronAlgo::InputTagsConfiguration inputCfg_ ;
    GsfElectronAlgo::StrategyConfiguration strategyCfg_ ;
    const GsfElectronAlgo::CutsConfiguration cutsCfg_ ;
    const GsfElectronAlgo::CutsConfiguration cutsCfgPflow_ ;
    ElectronHcalHelper::Configuration hcalCfg_ ;
    ElectronHcalHelper::Configuration hcalCfgPflow_ ;

    // used to make some provenance checks
    edm::EDGetTokenT<edm::ValueMap<float>> pfMVA_;

  private :

    bool isPreselected( reco::GsfElectron const& ele ) const ;
    void setAmbiguityData( reco::GsfElectronCollection & electrons, edm::Event const& event, bool ignoreNotPreselected = true ) const ;

    // check expected configuration of previous modules
    bool ecalSeedingParametersChecked_ ;
    void checkEcalSeedingParameters( edm::ParameterSet const & ) ;
    edm::OrphanHandle<reco::GsfElectronCollection> orphanHandle_;

    const edm::EDPutTokenT<reco::GsfElectronCollection> electronPutToken_;
 } ;

#endif
