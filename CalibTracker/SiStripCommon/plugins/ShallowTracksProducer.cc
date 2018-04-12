#include "CalibTracker/SiStripCommon/interface/ShallowTracksProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "boost/foreach.hpp"

ShallowTracksProducer::ShallowTracksProducer(const edm::ParameterSet& iConfig)
  :  tracks_token_( consumes<edm::View<reco::Track> >( iConfig.getParameter<edm::InputTag>("Tracks") )),
     Prefix       ( iConfig.getParameter<std::string>("Prefix")    ),
     Suffix       ( iConfig.getParameter<std::string>("Suffix")    )
{
  produces <unsigned int>               ( Prefix + "number"    + Suffix );
  produces <std::vector<float> >       ( Prefix + "chi2"      + Suffix );
  produces <std::vector<float> >       ( Prefix + "ndof"      + Suffix );
  produces <std::vector<float> >       ( Prefix + "chi2ndof"  + Suffix );
  produces <std::vector<float> >        ( Prefix + "charge"    + Suffix );
  produces <std::vector<float> >        ( Prefix + "momentum"  + Suffix );
  produces <std::vector<float> >        ( Prefix + "pt"        + Suffix );
  produces <std::vector<float> >        ( Prefix + "pterr"     + Suffix );
  produces <std::vector<unsigned int> > ( Prefix + "hitsvalid" + Suffix );
  produces <std::vector<unsigned int> > ( Prefix + "hitslost"  + Suffix );
  produces <std::vector<float> >       ( Prefix + "theta"     + Suffix );
  produces <std::vector<float> >       ( Prefix + "thetaerr"  + Suffix );
  produces <std::vector<float> >       ( Prefix + "phi"       + Suffix );
  produces <std::vector<float> >       ( Prefix + "phierr"    + Suffix );
  produces <std::vector<float> >       ( Prefix + "eta"       + Suffix );
  produces <std::vector<float> >       ( Prefix + "etaerr"    + Suffix );
  produces <std::vector<float> >       ( Prefix + "dxy"       + Suffix );
  produces <std::vector<float> >       ( Prefix + "dxyerr"    + Suffix );
  produces <std::vector<float> >       ( Prefix + "dsz"       + Suffix );
  produces <std::vector<float> >       ( Prefix + "dszerr"    + Suffix );
  produces <std::vector<float> >       ( Prefix + "qoverp"    + Suffix );
  produces <std::vector<float> >       ( Prefix + "qoverperr" + Suffix );
  produces <std::vector<float> >       ( Prefix + "vx"        + Suffix );
  produces <std::vector<float> >       ( Prefix + "vy"        + Suffix );
  produces <std::vector<float> >       ( Prefix + "vz"        + Suffix );
  produces <std::vector<int> >          ( Prefix + "algo"        + Suffix );
}

void ShallowTracksProducer::
produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto       number      = std::make_unique<unsigned int>(0);
  auto       chi2        = std::make_unique<std::vector<float>>();
  auto       ndof        = std::make_unique<std::vector<float>>();
  auto       chi2ndof    = std::make_unique<std::vector<float>>();
  auto       charge      = std::make_unique<std::vector<float>>();
  auto       momentum    = std::make_unique<std::vector<float>>();
  auto       pt          = std::make_unique<std::vector<float>>();
  auto       pterr       = std::make_unique<std::vector<float>>();
  auto       hitsvalid   = std::make_unique<std::vector<unsigned int>>();
  auto       hitslost    = std::make_unique<std::vector<unsigned int>>();
  auto       theta       = std::make_unique<std::vector<float>>();
  auto       thetaerr    = std::make_unique<std::vector<float>>();
  auto       phi         = std::make_unique<std::vector<float>>();
  auto       phierr      = std::make_unique<std::vector<float>>();
  auto       eta         = std::make_unique<std::vector<float>>();
  auto       etaerr      = std::make_unique<std::vector<float>>();
  auto       dxy         = std::make_unique<std::vector<float>>();
  auto       dxyerr      = std::make_unique<std::vector<float>>();
  auto       dsz         = std::make_unique<std::vector<float>>();
  auto       dszerr      = std::make_unique<std::vector<float>>();
  auto       qoverp      = std::make_unique<std::vector<float>>();
  auto       qoverperr   = std::make_unique<std::vector<float>>();
  auto       vx          = std::make_unique<std::vector<float>>();
  auto       vy          = std::make_unique<std::vector<float>>();
  auto       vz          = std::make_unique<std::vector<float>>();
  auto       algo        = std::make_unique<std::vector<int>>();

  edm::Handle<edm::View<reco::Track> > tracks;  iEvent.getByToken(tracks_token_, tracks);

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

  iEvent.put(std::move(number),       Prefix + "number"     + Suffix );
  iEvent.put(std::move(chi2),         Prefix + "chi2"       + Suffix );
  iEvent.put(std::move(ndof),         Prefix + "ndof"       + Suffix );
  iEvent.put(std::move(chi2ndof),     Prefix + "chi2ndof"   + Suffix );
  iEvent.put(std::move(charge),       Prefix + "charge"     + Suffix );
  iEvent.put(std::move(momentum),     Prefix + "momentum"   + Suffix );
  iEvent.put(std::move(pt),           Prefix + "pt"         + Suffix );
  iEvent.put(std::move(pterr),        Prefix + "pterr"      + Suffix );
  iEvent.put(std::move(hitsvalid),    Prefix + "hitsvalid"  + Suffix );
  iEvent.put(std::move(hitslost),     Prefix + "hitslost"   + Suffix );
  iEvent.put(std::move(theta),        Prefix + "theta"      + Suffix );
  iEvent.put(std::move(thetaerr),     Prefix + "thetaerr"   + Suffix );
  iEvent.put(std::move(phi),          Prefix + "phi"        + Suffix );
  iEvent.put(std::move(phierr),       Prefix + "phierr"     + Suffix );
  iEvent.put(std::move(eta),          Prefix + "eta"        + Suffix );
  iEvent.put(std::move(etaerr),       Prefix + "etaerr"     + Suffix );
  iEvent.put(std::move(dxy),          Prefix + "dxy"        + Suffix );
  iEvent.put(std::move(dxyerr),       Prefix + "dxyerr"     + Suffix );
  iEvent.put(std::move(dsz),          Prefix + "dsz"        + Suffix );
  iEvent.put(std::move(dszerr),       Prefix + "dszerr"     + Suffix );
  iEvent.put(std::move(qoverp),       Prefix + "qoverp"     + Suffix );
  iEvent.put(std::move(qoverperr),    Prefix + "qoverperr"  + Suffix );
  iEvent.put(std::move(vx),           Prefix + "vx"         + Suffix );
  iEvent.put(std::move(vy),           Prefix + "vy"         + Suffix );
  iEvent.put(std::move(vz),           Prefix + "vz"         + Suffix );
  iEvent.put(std::move(algo),         Prefix + "algo"       + Suffix );

}
