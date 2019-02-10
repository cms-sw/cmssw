#ifndef RecoEgamma_EgammaElectronProducers_LowPtGsfElectronIDProducer_h
#define RecoEgamma_EgammaElectronProducers_LowPtGsfElectronIDProducer_h

#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "RecoEgamma/EgammaElectronProducers/interface/LowPtGsfElectronIDHeavyObjectCache.h"
#include <string>
#include <vector>

class LowPtGsfElectronIDProducer final : 
public edm::stream::EDProducer< edm::GlobalCache<lowptgsfeleid::HeavyObjectCache> > {
  
 public:
  
  explicit LowPtGsfElectronIDProducer( const edm::ParameterSet&,
				       const lowptgsfeleid::HeavyObjectCache* );
  
  ~LowPtGsfElectronIDProducer() override;
  
  static std::unique_ptr<lowptgsfeleid::HeavyObjectCache> 
    initializeGlobalCache( const edm::ParameterSet& conf ) {
    return std::make_unique<lowptgsfeleid::HeavyObjectCache>(lowptgsfeleid::HeavyObjectCache(conf));
  }
  
  static void globalEndJob( lowptgsfeleid::HeavyObjectCache const* ) {}

  void produce( edm::Event&, const edm::EventSetup& ) override;
  
  static void fillDescriptions( edm::ConfigurationDescriptions& );

 private:
  
  const edm::EDGetTokenT<reco::GsfElectronCollection> gsfElectrons_;
  const edm::EDGetTokenT<double> rho_;
  const std::vector<std::string> names_;
  const bool passThrough_;
  const double minPtThreshold_;
  const double maxPtThreshold_;

};

#endif // RecoEgamma_EgammaElectronProducers_LowPtGsfElectronIDProducer_h

