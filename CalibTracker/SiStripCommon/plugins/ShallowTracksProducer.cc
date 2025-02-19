#include "CalibTracker/SiStripCommon/interface/ShallowTracksProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "boost/foreach.hpp"

ShallowTracksProducer::ShallowTracksProducer(const edm::ParameterSet& iConfig)
  :  theTracksLabel( iConfig.getParameter<edm::InputTag>("Tracks") ),
     Prefix       ( iConfig.getParameter<std::string>("Prefix")    ),
     Suffix       ( iConfig.getParameter<std::string>("Suffix")    )
{
  produces <unsigned int>               ( Prefix + "number"    + Suffix );
  produces <std::vector<double> >       ( Prefix + "chi2"      + Suffix );
  produces <std::vector<double> >       ( Prefix + "ndof"      + Suffix );
  produces <std::vector<double> >       ( Prefix + "chi2ndof"  + Suffix );
  produces <std::vector<float> >        ( Prefix + "charge"    + Suffix );
  produces <std::vector<float> >        ( Prefix + "momentum"  + Suffix );
  produces <std::vector<float> >        ( Prefix + "pt"        + Suffix );
  produces <std::vector<float> >        ( Prefix + "pterr"     + Suffix );
  produces <std::vector<unsigned int> > ( Prefix + "hitsvalid" + Suffix );
  produces <std::vector<unsigned int> > ( Prefix + "hitslost"  + Suffix );
  produces <std::vector<double> >       ( Prefix + "theta"     + Suffix );
  produces <std::vector<double> >       ( Prefix + "thetaerr"  + Suffix );
  produces <std::vector<double> >       ( Prefix + "phi"       + Suffix );
  produces <std::vector<double> >       ( Prefix + "phierr"    + Suffix );
  produces <std::vector<double> >       ( Prefix + "eta"       + Suffix );
  produces <std::vector<double> >       ( Prefix + "etaerr"    + Suffix );
  produces <std::vector<double> >       ( Prefix + "dxy"       + Suffix );
  produces <std::vector<double> >       ( Prefix + "dxyerr"    + Suffix );
  produces <std::vector<double> >       ( Prefix + "dsz"       + Suffix );
  produces <std::vector<double> >       ( Prefix + "dszerr"    + Suffix );
  produces <std::vector<double> >       ( Prefix + "qoverp"    + Suffix );
  produces <std::vector<double> >       ( Prefix + "qoverperr" + Suffix );
  produces <std::vector<double> >       ( Prefix + "vx"        + Suffix );
  produces <std::vector<double> >       ( Prefix + "vy"        + Suffix );
  produces <std::vector<double> >       ( Prefix + "vz"        + Suffix );
  produces <std::vector<int> >          ( Prefix + "algo"        + Suffix );
}

void ShallowTracksProducer::
produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  std::auto_ptr<unsigned int>               number      ( new unsigned int(0)             );
  std::auto_ptr<std::vector<double> >       chi2        ( new std::vector<double>()       );
  std::auto_ptr<std::vector<double> >       ndof        ( new std::vector<double>()       );
  std::auto_ptr<std::vector<double> >       chi2ndof    ( new std::vector<double>()       );
  std::auto_ptr<std::vector<float> >        charge      ( new std::vector<float>()        );
  std::auto_ptr<std::vector<float> >        momentum    ( new std::vector<float>()        );
  std::auto_ptr<std::vector<float> >        pt          ( new std::vector<float>()        );
  std::auto_ptr<std::vector<float> >        pterr       ( new std::vector<float>()        );
  std::auto_ptr<std::vector<unsigned int> > hitsvalid   ( new std::vector<unsigned int>() );
  std::auto_ptr<std::vector<unsigned int> > hitslost    ( new std::vector<unsigned int>() );
  std::auto_ptr<std::vector<double> >       theta       ( new std::vector<double>()       );
  std::auto_ptr<std::vector<double> >       thetaerr    ( new std::vector<double>()       );
  std::auto_ptr<std::vector<double> >       phi         ( new std::vector<double>()       );
  std::auto_ptr<std::vector<double> >       phierr      ( new std::vector<double>()       );
  std::auto_ptr<std::vector<double> >       eta         ( new std::vector<double>()       );
  std::auto_ptr<std::vector<double> >       etaerr      ( new std::vector<double>()       );
  std::auto_ptr<std::vector<double> >       dxy         ( new std::vector<double>()       );
  std::auto_ptr<std::vector<double> >       dxyerr      ( new std::vector<double>()       );
  std::auto_ptr<std::vector<double> >       dsz         ( new std::vector<double>()       );
  std::auto_ptr<std::vector<double> >       dszerr      ( new std::vector<double>()       );
  std::auto_ptr<std::vector<double> >       qoverp      ( new std::vector<double>()       );
  std::auto_ptr<std::vector<double> >       qoverperr   ( new std::vector<double>()       );
  std::auto_ptr<std::vector<double> >       vx          ( new std::vector<double>()       );
  std::auto_ptr<std::vector<double> >       vy          ( new std::vector<double>()       );
  std::auto_ptr<std::vector<double> >       vz          ( new std::vector<double>()       );
  std::auto_ptr<std::vector<int> >          algo        ( new std::vector<int>() );

  edm::Handle<edm::View<reco::Track> > tracks;  iEvent.getByLabel(theTracksLabel, tracks);
  
  *number = tracks->size();
  BOOST_FOREACH( const reco::Track track, *tracks) {
    chi2->push_back(      track.chi2()              );
    ndof->push_back(      track.ndof()              );
    chi2ndof->push_back(  track.chi2()/track.ndof() );
    charge->push_back(    track.charge()            );
    momentum->push_back(  track.p()                 );
    pt->push_back(        track.pt()                );
    pterr->push_back(     track.ptError()           );
    hitsvalid->push_back( track.numberOfValidHits() );
    hitslost->push_back(  track.numberOfLostHits()  );
    theta->push_back(     track.theta()             );
    thetaerr->push_back(  track.thetaError()        );
    phi->push_back(       track.phi()               );
    phierr->push_back(    track.phiError()          );
    eta->push_back(       track.eta()               );
    etaerr->push_back(    track.etaError()          );
    dxy->push_back(       track.dxy()               );
    dxyerr->push_back(    track.dxyError()          );
    dsz->push_back(       track.dsz()               );
    dszerr->push_back(    track.dszError()          );
    qoverp->push_back(    track.qoverp()            );
    qoverperr->push_back( track.qoverpError()       );
    vx->push_back(        track.vx()                );
    vy->push_back(        track.vy()                );
    vz->push_back(        track.vz()                );
    algo->push_back(      (int) track.algo()              );
  }			  
  			  
  iEvent.put(number,       Prefix + "number"     + Suffix );
  iEvent.put(chi2,         Prefix + "chi2"       + Suffix );
  iEvent.put(ndof,         Prefix + "ndof"       + Suffix );
  iEvent.put(chi2ndof,     Prefix + "chi2ndof"   + Suffix );
  iEvent.put(charge,       Prefix + "charge"     + Suffix );
  iEvent.put(momentum,     Prefix + "momentum"   + Suffix );
  iEvent.put(pt,           Prefix + "pt"         + Suffix );
  iEvent.put(pterr,        Prefix + "pterr"      + Suffix );
  iEvent.put(hitsvalid,    Prefix + "hitsvalid"  + Suffix );
  iEvent.put(hitslost,     Prefix + "hitslost"   + Suffix );
  iEvent.put(theta,        Prefix + "theta"      + Suffix );
  iEvent.put(thetaerr,     Prefix + "thetaerr"   + Suffix );
  iEvent.put(phi,          Prefix + "phi"        + Suffix );
  iEvent.put(phierr,       Prefix + "phierr"     + Suffix );
  iEvent.put(eta,          Prefix + "eta"        + Suffix );
  iEvent.put(etaerr,       Prefix + "etaerr"     + Suffix );
  iEvent.put(dxy,          Prefix + "dxy"        + Suffix );
  iEvent.put(dxyerr,       Prefix + "dxyerr"     + Suffix );
  iEvent.put(dsz,          Prefix + "dsz"        + Suffix );
  iEvent.put(dszerr,       Prefix + "dszerr"     + Suffix );
  iEvent.put(qoverp,       Prefix + "qoverp"     + Suffix );
  iEvent.put(qoverperr,    Prefix + "qoverperr"  + Suffix );
  iEvent.put(vx,           Prefix + "vx"         + Suffix );
  iEvent.put(vy,           Prefix + "vy"         + Suffix );
  iEvent.put(vz,           Prefix + "vz"         + Suffix );
  iEvent.put(algo,         Prefix + "algo"         + Suffix );

}

