#ifndef RecoEgamma_EgammaElectronProducers_LowPtGsfElectronProducer_h
#define RecoEgamma_EgammaElectronProducers_LowPtGsfElectronProducer_h

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "RecoEgamma/EgammaElectronProducers/plugins/GsfElectronBaseProducer.h"

class LowPtGsfElectronProducer : public GsfElectronBaseProducer {
  
 public:
  
  explicit LowPtGsfElectronProducer( const edm::ParameterSet&, 
				     const gsfAlgoHelpers::HeavyObjectCache* );

  ~LowPtGsfElectronProducer() override;

  void beginEvent( edm::Event&, const edm::EventSetup& );

  void produce( edm::Event&, const edm::EventSetup& ) override;

  //static void fillDescriptions( edm::ConfigurationDescriptions& );

 private:

};

#endif // RecoEgamma_EgammaElectronProducers_LowPtGsfElectronProducer_h
