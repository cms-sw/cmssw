
#ifndef GEDGsfElectronProducer_h
#define GEDGsfElectronProducer_h

#include "GsfElectronBaseProducer.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

class GEDGsfElectronProducer : public GsfElectronBaseProducer
 {
  public:

    //static void fillDescriptions( edm::ConfigurationDescriptions & ) ;

    explicit GEDGsfElectronProducer( const edm::ParameterSet & ) ;
    virtual ~GEDGsfElectronProducer() ;
    virtual void produce( edm::Event &, const edm::EventSetup & ) ;

 private:
    edm::EDGetTokenT<reco::PFCandidateCollection> egmPFCandidateCollection_;
    std::string outputValueMapLabel_;

 private:
    void matchWithPFCandidates(edm::Event & event, edm::ValueMap<reco::GsfElectronRef>::Filler & filler);

 } ;

#endif
