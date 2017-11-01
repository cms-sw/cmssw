#ifndef GEDGsfElectronCoreProducer_h
#define GEDGsfElectronCoreProducer_h

#include "GsfElectronCoreBaseProducer.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronCoreFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

class GEDGsfElectronCoreProducer : public GsfElectronCoreBaseProducer
 {
  public:

    static void fillDescriptions( edm::ConfigurationDescriptions & ) ;

    explicit GEDGsfElectronCoreProducer( const edm::ParameterSet & conf ) ;
    ~GEDGsfElectronCoreProducer() override ;
    void produce( edm::Event&, const edm::EventSetup & ) override ;

  private:

    void produceElectronCore( const reco::PFCandidate & pfCandidate, reco::GsfElectronCoreCollection * electrons ) ;    

    edm::EDGetTokenT<reco::PFCandidateCollection> gedEMUnbiasedTag_ ;
 } ;

#endif
