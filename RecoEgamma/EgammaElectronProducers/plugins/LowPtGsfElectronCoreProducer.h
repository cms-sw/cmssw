#ifndef RecoEgamma_EgammaElectronProducers_LowPtGsfElectronCoreProducer_h
#define RecoEgamma_EgammaElectronProducers_LowPtGsfElectronCoreProducer_h

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "RecoEgamma/EgammaElectronProducers/plugins/GsfElectronCoreBaseProducer.h"

class LowPtGsfElectronCoreProducer : public GsfElectronCoreBaseProducer {

 public:
  
  explicit LowPtGsfElectronCoreProducer( const edm::ParameterSet& conf );
  
  ~LowPtGsfElectronCoreProducer() override;
  
  void produce( edm::Event&, const edm::EventSetup& ) override;
  
  static void fillDescriptions( edm::ConfigurationDescriptions& );

 private:

  edm::EDGetTokenT<reco::SuperClusterCollection> superClusters_;
  edm::EDGetTokenT< edm::ValueMap<reco::SuperClusterRef> > superClusterRefs_;

};

#endif // RecoEgamma_EgammaElectronProducers_LowPtGsfElectronCoreProducer_h

