
#ifndef GsfElectronFull5x5Filler_h
#define GsfElectronFull5x5Filler_h

#include "GsfElectronBaseProducer.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include <map>

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

class GsfElectronFull5x5Filler : public edm::stream::EDProducer<>
 {
  public:

    //static void fillDescriptions( edm::ConfigurationDescriptions & ) ;

    explicit GsfElectronFull5x5Filler( const edm::ParameterSet & ) ;
    ~GsfElectronFull5x5Filler() override ;
    void produce( edm::Event &, const edm::EventSetup & ) override ;
    void calculateShowerShape_full5x5(const reco::SuperClusterRef & theClus, bool pflow, reco::GsfElectron::ShowerShape & showerShape);

    void beginLuminosityBlock(edm::LuminosityBlock const&, 
			      edm::EventSetup const&) override;

 private:
    edm::EDGetTokenT<reco::GsfElectronCollection> _source;
    edm::EDGetTokenT<EcalRecHitCollection> _ebRecHitsToken, _eeRecHitsToken;
    std::unique_ptr<ElectronHcalHelper> _hcalHelper, _hcalHelperPflow ;
    edm::Handle<EcalRecHitCollection> _ebRecHits;
    edm::Handle<EcalRecHitCollection> _eeRecHits;
    const CaloTopology * _topology;
    const CaloGeometry * _geometry;
    
 private:
 };

#endif
