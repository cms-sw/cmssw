#ifndef GsfElectronCoreBaseProducer_h
#define GsfElectronCoreBaseProducer_h

//
// Package:         RecoEgamma/EgammaElectronProducers
// Class:           GsfElectronCoreBaseProducer
//
// Description:


#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

namespace reco
 {
  class GsfElectronCore ;
 }

namespace edm
 {
  class ParameterSet ;
  class ConfigurationDescriptions ;
 }

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

class GsfElectronCoreBaseProducer : public edm::stream::EDProducer<>
 {
  public:

    static void fillDescription( edm::ParameterSetDescription & ) ;

    explicit GsfElectronCoreBaseProducer( const edm::ParameterSet & conf ) ;
    virtual ~GsfElectronCoreBaseProducer() ;


  protected:

    // to be called by derived producers at the beginning of each new event
    void initEvent( edm::Event & event, const edm::EventSetup & setup ) ;
    edm::Handle<reco::GsfPFRecTrackCollection> gsfPfRecTracksH_ ;
    edm::Handle<reco::GsfTrackCollection> gsfTracksH_ ;
    edm::Handle<reco::TrackCollection> ctfTracksH_ ;
    bool useGsfPfRecTracks_ ;

    void fillElectronCore( reco::GsfElectronCore * ) ;

  private:

    edm::EDGetTokenT<reco::GsfPFRecTrackCollection> gsfPfRecTracksTag_ ;
    edm::EDGetTokenT<reco::GsfTrackCollection> gsfTracksTag_ ;
    edm::EDGetTokenT<reco::TrackCollection> ctfTracksTag_ ;

    // From Puneeth Kalavase : returns the CTF track that has the highest fraction
    // of shared hits in Pixels and the inner strip tracker with the electron Track
    std::pair<reco::TrackRef,float> getCtfTrackRef
     ( const reco::GsfTrackRef & ) ;

 } ;


#endif
