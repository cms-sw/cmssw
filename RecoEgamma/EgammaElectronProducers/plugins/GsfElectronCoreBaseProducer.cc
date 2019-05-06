#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronCoreFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronCore.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/GsfElectronTools.h"

#include "GsfElectronCoreBaseProducer.h"

using namespace reco ;

void GsfElectronCoreBaseProducer::fillDescription( edm::ParameterSetDescription & desc )
 {
  desc.add<edm::InputTag>("gsfPfRecTracks",edm::InputTag("pfTrackElec")) ;
  desc.add<edm::InputTag>("gsfTracks",edm::InputTag("electronGsfTracks")) ;
  desc.add<edm::InputTag>("ctfTracks",edm::InputTag("generalTracks")) ;
  desc.add<bool>("useGsfPfRecTracks",true) ;
 }

GsfElectronCoreBaseProducer::GsfElectronCoreBaseProducer( const edm::ParameterSet & config )
 {
  produces<GsfElectronCoreCollection>() ;
  gsfPfRecTracksTag_ = mayConsume<reco::GsfPFRecTrackCollection>(config.getParameter<edm::InputTag>("gsfPfRecTracks")) ;
  gsfTracksTag_ = consumes<reco::GsfTrackCollection>(config.getParameter<edm::InputTag>("gsfTracks"));
  ctfTracksTag_ = consumes<reco::TrackCollection>(config.getParameter<edm::InputTag>("ctfTracks"));
  useGsfPfRecTracks_ = config.getParameter<bool>("useGsfPfRecTracks") ;
 }

GsfElectronCoreBaseProducer::~GsfElectronCoreBaseProducer()
 {}


//=======================================================================================
// For derived producers
//=======================================================================================

// to be called at the beginning of each new event
void GsfElectronCoreBaseProducer::initEvent( edm::Event & event, const edm::EventSetup & setup )
 {
  if (useGsfPfRecTracks_)
   { event.getByToken(gsfPfRecTracksTag_,gsfPfRecTracksH_) ; }
  event.getByToken(gsfTracksTag_,gsfTracksH_) ;
  event.getByToken(ctfTracksTag_,ctfTracksH_) ;
 }

void GsfElectronCoreBaseProducer::fillElectronCore( reco::GsfElectronCore * eleCore )
 {
  const GsfTrackRef & gsfTrackRef = eleCore->gsfTrack() ;

  std::pair<TrackRef,float> ctfpair = gsfElectronTools::getClosestCtfToGsf(gsfTrackRef,ctfTracksH_) ;
  eleCore->setCtfTrack(ctfpair.first,ctfpair.second) ;
 }
