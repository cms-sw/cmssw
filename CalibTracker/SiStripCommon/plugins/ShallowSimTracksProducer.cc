#include "CalibTracker/SiStripCommon/interface/ShallowSimTracksProducer.h"
#include "CalibTracker/SiStripCommon/interface/ShallowTools.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"
#include "DataFormats/RecoCandidate/interface/TrackAssociation.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

ShallowSimTracksProducer::ShallowSimTracksProducer(const edm::ParameterSet& conf) 
  : Prefix( conf.getParameter<std::string>("Prefix") ),
    Suffix( conf.getParameter<std::string>("Suffix") ),
    trackingParticles_tag( conf.getParameter<edm::InputTag>("TrackingParticles")),
    associator_tag( conf.getParameter<edm::InputTag>("Associator")),
    tracks_tag( conf.getParameter<edm::InputTag>("Tracks"))
{
  produces <std::vector<unsigned> >     ( Prefix + "multi"      + Suffix );
  produces <std::vector<int> >          ( Prefix + "type"      + Suffix );
  produces <std::vector<float> >        ( Prefix + "charge"    + Suffix );
  produces <std::vector<float> >        ( Prefix + "momentum"  + Suffix );
  produces <std::vector<float> >        ( Prefix + "pt"        + Suffix );
  produces <std::vector<double> >       ( Prefix + "theta"     + Suffix );
  produces <std::vector<double> >       ( Prefix + "phi"       + Suffix );
  produces <std::vector<double> >       ( Prefix + "eta"       + Suffix );
  produces <std::vector<double> >       ( Prefix + "qoverp"    + Suffix );
  produces <std::vector<double> >       ( Prefix + "vx"        + Suffix );
  produces <std::vector<double> >       ( Prefix + "vy"        + Suffix );
  produces <std::vector<double> >       ( Prefix + "vz"        + Suffix );
}


void ShallowSimTracksProducer::
produce(edm::Event& event, const edm::EventSetup& setup) {

  edm::Handle<edm::View<reco::Track> >                     tracks ;   event.getByLabel( tracks_tag, tracks);
  edm::Handle<TrackingParticleCollection>       trackingParticles ;   event.getByLabel( trackingParticles_tag, trackingParticles );  
  edm::Handle<reco::TrackToTrackingParticleAssociator> associator ;   event.getByLabel( associator_tag, associator);

  unsigned size = tracks->size();
  std::auto_ptr<std::vector<unsigned> > multi        ( new std::vector<unsigned>(size,    0));
  std::auto_ptr<std::vector<int> >      type         ( new std::vector<int>     (size,    0));
  std::auto_ptr<std::vector<float> >    charge       ( new std::vector<float>   (size,    0));
  std::auto_ptr<std::vector<float> >    momentum     ( new std::vector<float>   (size,   -1));
  std::auto_ptr<std::vector<float> >    pt           ( new std::vector<float>   (size,   -1));
  std::auto_ptr<std::vector<double> >   theta        ( new std::vector<double>  (size,-1000));
  std::auto_ptr<std::vector<double> >   phi          ( new std::vector<double>  (size,-1000));
  std::auto_ptr<std::vector<double> >   eta          ( new std::vector<double>  (size,-1000));
  std::auto_ptr<std::vector<double> >   dxy          ( new std::vector<double>  (size,-1000));
  std::auto_ptr<std::vector<double> >   dsz          ( new std::vector<double>  (size,-1000));
  std::auto_ptr<std::vector<double> >   qoverp       ( new std::vector<double>  (size,-1000));
  std::auto_ptr<std::vector<double> >   vx           ( new std::vector<double>  (size,-1000));
  std::auto_ptr<std::vector<double> >   vy           ( new std::vector<double>  (size,-1000));
  std::auto_ptr<std::vector<double> >   vz           ( new std::vector<double>  (size,-1000));

  reco::RecoToSimCollection associations = associator->associateRecoToSim( tracks, trackingParticles);
  
  for( reco::RecoToSimCollection::const_iterator association = associations.begin(); 
       association != associations.end(); association++) {

    const reco::Track* track = association->key.get();
    const int matches        = association->val.size();
    if(matches>0) {
      const TrackingParticle* tparticle = association->val[0].first.get();
      unsigned i = shallow::findTrackIndex(tracks, track);

      multi->at(i) = matches;
      type->at(i)  = tparticle->pdgId();
      charge->at(i)= tparticle->charge();
      momentum->at(i)=tparticle->p() ;
      pt->at(i) = tparticle->pt()    ;
      theta->at(i) = tparticle->theta() ;
      phi->at(i)   = tparticle->phi()   ;
      eta->at(i)   = tparticle->eta()   ;
      qoverp->at(i)= tparticle->charge()/tparticle->p();

      const TrackingVertex* tvertex = tparticle->parentVertex().get();
      vx->at(i) = tvertex->position().x();
      vy->at(i) = tvertex->position().y();
      vz->at(i) = tvertex->position().z();
    }
  }
  
  event.put(  multi    ,Prefix + "multi"     + Suffix );
  event.put(  type     ,Prefix + "type"      + Suffix );
  event.put(  charge   ,Prefix + "charge"    + Suffix );
  event.put(  momentum ,Prefix + "momentum"  + Suffix );
  event.put(  pt       ,Prefix + "pt"        + Suffix );
  event.put(  theta    ,Prefix + "theta"     + Suffix );
  event.put(  phi      ,Prefix + "phi"       + Suffix );
  event.put(  eta      ,Prefix + "eta"       + Suffix );
  event.put(  qoverp   ,Prefix + "qoverp"    + Suffix );
  event.put(  vx       ,Prefix + "vx"        + Suffix );
  event.put(  vy       ,Prefix + "vy"        + Suffix );
  event.put(  vz       ,Prefix + "vz"        + Suffix );
  
}
