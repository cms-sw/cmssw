#include "ShallowTracksProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"

ShallowTracksProducer::ShallowTracksProducer(const edm::ParameterSet& iConfig)
    : tracks_token_(consumes<edm::View<reco::Track>>(iConfig.getParameter<edm::InputTag>("Tracks"))),
      Prefix(iConfig.getParameter<std::string>("Prefix")),
      Suffix(iConfig.getParameter<std::string>("Suffix")),
      numberPut_(produces<unsigned int>(Prefix + "number" + Suffix)),
      chi2Put_(produces<std::vector<double>>(Prefix + "chi2" + Suffix)),
      ndofPut_(produces<std::vector<double>>(Prefix + "ndof" + Suffix)),
      chi2ndofPut_(produces<std::vector<double>>(Prefix + "chi2ndof" + Suffix)),
      chargePut_(produces<std::vector<float>>(Prefix + "charge" + Suffix)),
      momentumPut_(produces<std::vector<float>>(Prefix + "momentum" + Suffix)),
      ptPut_(produces<std::vector<float>>(Prefix + "pt" + Suffix)),
      pterrPut_(produces<std::vector<float>>(Prefix + "pterr" + Suffix)),
      hitsvalidPut_(produces<std::vector<unsigned int>>(Prefix + "hitsvalid" + Suffix)),
      hitslostPut_(produces<std::vector<unsigned int>>(Prefix + "hitslost" + Suffix)),
      thetaPut_(produces<std::vector<double>>(Prefix + "theta" + Suffix)),
      thetaerrPut_(produces<std::vector<double>>(Prefix + "thetaerr" + Suffix)),
      phiPut_(produces<std::vector<double>>(Prefix + "phi" + Suffix)),
      phierrPut_(produces<std::vector<double>>(Prefix + "phierr" + Suffix)),
      etaPut_(produces<std::vector<double>>(Prefix + "eta" + Suffix)),
      etaerrPut_(produces<std::vector<double>>(Prefix + "etaerr" + Suffix)),
      dxyPut_(produces<std::vector<double>>(Prefix + "dxy" + Suffix)),
      dxyerrPut_(produces<std::vector<double>>(Prefix + "dxyerr" + Suffix)),
      dszPut_(produces<std::vector<double>>(Prefix + "dsz" + Suffix)),
      dszerrPut_(produces<std::vector<double>>(Prefix + "dszerr" + Suffix)),
      qoverpPut_(produces<std::vector<double>>(Prefix + "qoverp" + Suffix)),
      qoverperrPut_(produces<std::vector<double>>(Prefix + "qoverperr" + Suffix)),
      vxPut_(produces<std::vector<double>>(Prefix + "vx" + Suffix)),
      vyPut_(produces<std::vector<double>>(Prefix + "vy" + Suffix)),
      vzPut_(produces<std::vector<double>>(Prefix + "vz" + Suffix)),
      algoPut_(produces<std::vector<int>>(Prefix + "algo" + Suffix)) {}

void ShallowTracksProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  edm::Handle<edm::View<reco::Track>> tracks;
  iEvent.getByToken(tracks_token_, tracks);

  unsigned int number = tracks->size();
  std::vector<double> chi2;
  chi2.reserve(number);
  std::vector<double> ndof;
  ndof.reserve(number);
  std::vector<double> chi2ndof;
  chi2ndof.reserve(number);
  std::vector<float> charge;
  charge.reserve(number);
  std::vector<float> momentum;
  momentum.reserve(number);
  std::vector<float> pt;
  pt.reserve(number);
  std::vector<float> pterr;
  pterr.reserve(number);
  std::vector<unsigned int> hitsvalid;
  hitsvalid.reserve(number);
  std::vector<unsigned int> hitslost;
  hitslost.reserve(number);
  std::vector<double> theta;
  theta.reserve(number);
  std::vector<double> thetaerr;
  thetaerr.reserve(number);
  std::vector<double> phi;
  phi.reserve(number);
  std::vector<double> phierr;
  phierr.reserve(number);
  std::vector<double> eta;
  eta.reserve(number);
  std::vector<double> etaerr;
  etaerr.reserve(number);
  std::vector<double> dxy;
  dxy.reserve(number);
  std::vector<double> dxyerr;
  dxyerr.reserve(number);
  std::vector<double> dsz;
  dsz.reserve(number);
  std::vector<double> dszerr;
  dszerr.reserve(number);
  std::vector<double> qoverp;
  qoverp.reserve(number);
  std::vector<double> qoverperr;
  qoverperr.reserve(number);
  std::vector<double> vx;
  vx.reserve(number);
  std::vector<double> vy;
  vy.reserve(number);
  std::vector<double> vz;
  vz.reserve(number);
  std::vector<int> algo;
  algo.reserve(number);

  for (auto const& track : *tracks) {
    chi2.push_back(track.chi2());
    ndof.push_back(track.ndof());
    chi2ndof.push_back(track.chi2() / track.ndof());
    charge.push_back(track.charge());
    momentum.push_back(track.p());
    pt.push_back(track.pt());
    pterr.push_back(track.ptError());
    hitsvalid.push_back(track.numberOfValidHits());
    hitslost.push_back(track.numberOfLostHits());
    theta.push_back(track.theta());
    thetaerr.push_back(track.thetaError());
    phi.push_back(track.phi());
    phierr.push_back(track.phiError());
    eta.push_back(track.eta());
    etaerr.push_back(track.etaError());
    dxy.push_back(track.dxy());
    dxyerr.push_back(track.dxyError());
    dsz.push_back(track.dsz());
    dszerr.push_back(track.dszError());
    qoverp.push_back(track.qoverp());
    qoverperr.push_back(track.qoverpError());
    vx.push_back(track.vx());
    vy.push_back(track.vy());
    vz.push_back(track.vz());
    algo.push_back((int)track.algo());
  }

  iEvent.emplace(numberPut_, number);
  iEvent.emplace(chi2Put_, std::move(chi2));
  iEvent.emplace(ndofPut_, std::move(ndof));
  iEvent.emplace(chi2ndofPut_, std::move(chi2ndof));
  iEvent.emplace(chargePut_, std::move(charge));
  iEvent.emplace(momentumPut_, std::move(momentum));
  iEvent.emplace(ptPut_, std::move(pt));
  iEvent.emplace(pterrPut_, std::move(pterr));
  iEvent.emplace(hitsvalidPut_, std::move(hitsvalid));
  iEvent.emplace(hitslostPut_, std::move(hitslost));
  iEvent.emplace(thetaPut_, std::move(theta));
  iEvent.emplace(thetaerrPut_, std::move(thetaerr));
  iEvent.emplace(phiPut_, std::move(phi));
  iEvent.emplace(phierrPut_, std::move(phierr));
  iEvent.emplace(etaPut_, std::move(eta));
  iEvent.emplace(etaerrPut_, std::move(etaerr));
  iEvent.emplace(dxyPut_, std::move(dxy));
  iEvent.emplace(dxyerrPut_, std::move(dxyerr));
  iEvent.emplace(dszPut_, std::move(dsz));
  iEvent.emplace(dszerrPut_, std::move(dszerr));
  iEvent.emplace(qoverpPut_, std::move(qoverp));
  iEvent.emplace(qoverperrPut_, std::move(qoverperr));
  iEvent.emplace(vxPut_, std::move(vx));
  iEvent.emplace(vyPut_, std::move(vy));
  iEvent.emplace(vzPut_, std::move(vz));
  iEvent.emplace(algoPut_, std::move(algo));
}
