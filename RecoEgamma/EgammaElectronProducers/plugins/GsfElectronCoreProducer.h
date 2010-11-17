#ifndef GsfElectronCoreProducer_h
#define GsfElectronCoreProducer_h

//
// Package:         RecoEgamma/EgammaElectronProducers
// Class:           GsfElectronCoreProducer
//
// Description:


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"



class GsfElectronCoreProducer : public edm::EDProducer
 {
  public:

    explicit GsfElectronCoreProducer( const edm::ParameterSet & conf ) ;
    virtual ~GsfElectronCoreProducer() ;
    virtual void produce( edm::Event& e, const edm::EventSetup & c ) ;

  private:

    edm::InputTag gsfTracksTag_ ;
    edm::InputTag ctfTracksTag_;
    edm::InputTag pfSuperClustersTag_ ;
    edm::InputTag pfSuperClusterTrackMapTag_ ;

    // From Puneeth Kalavase : returns the CTF track that has the highest fraction
    // of shared hits in Pixels and the inner strip tracker with the electron Track
    std::pair<reco::TrackRef,float> getCtfTrackRef
     ( const reco::GsfTrackRef &, edm::Handle<reco::TrackCollection> ctfTracksH ) ;

 } ;


#endif
