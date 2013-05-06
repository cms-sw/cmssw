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
    virtual ~GEDGsfElectronCoreProducer() ;
    virtual void produce( edm::Event&, const edm::EventSetup & ) ;

  private:

    void produceElectronCore( const reco::PFCandidate & pfCandidate, reco::GsfElectronCoreCollection * electrons ) ;

    edm::Handle<reco::PFCandidateCollection> gedEMUnbiasedH_;

    edm::InputTag gedEMUnbiasedTag_ ;
 } ;

#endif
