#ifndef LowPtGsfElectronCoreProducer_h
#define LowPtGsfElectronCoreProducer_h

#include "DataFormats/EgammaCandidates/interface/GsfElectronCoreFwd.h"
#include "RecoEgamma/EgammaElectronProducers/plugins/GsfElectronCoreBaseProducer.h"

class LowPtGsfElectronCoreProducer : public GsfElectronCoreBaseProducer {

 public:
  
  explicit LowPtGsfElectronCoreProducer( const edm::ParameterSet& conf );
  
  ~LowPtGsfElectronCoreProducer() override;
  
  void produce( edm::Event&, const edm::EventSetup& ) override;
  
 private:
  
  void produceTrackerDrivenCore( const reco::GsfTrackRef& gsfTrackRef, 
				 reco::GsfElectronCoreCollection* electrons );
  
};

#endif // LowPtGsfElectronCoreProducer_h

