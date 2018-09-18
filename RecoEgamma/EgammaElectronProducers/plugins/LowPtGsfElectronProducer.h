#ifndef LowPtGsfElectronProducer_h
#define LowPtGsfElectronProducer_h

#include "RecoEgamma/EgammaElectronProducers/plugins/GsfElectronBaseProducer.h"

class LowPtGsfElectronProducer : public GsfElectronBaseProducer {
  
 public:
  
  explicit LowPtGsfElectronProducer( const edm::ParameterSet&, 
				     const gsfAlgoHelpers::HeavyObjectCache* );

  ~LowPtGsfElectronProducer() override;

  void produce( edm::Event&, const edm::EventSetup& ) override;

 private:

};

#endif // LowPtGsfElectronProducer_h
