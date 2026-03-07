#include "DataFormats/PatCandidates/interface/ScoutingDataHandling.h"
#include "DataFormats/Math/interface/Error.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/libminifloat.h"

float reduce_precision(float f, int mantissaPrecision = 10) {
  return MiniFloatConverter::reduceMantissaToNbitsRounding(f, mantissaPrecision);
}

reco::Track pat::makeRecoTrack(const Run3ScoutingTrack& sTrack) {
  reco::Track::Point v(sTrack.tk_vx(), sTrack.tk_vy(), sTrack.tk_vz());
  reco::Track::Vector p(math::RhoEtaPhiVector(sTrack.tk_pt(), sTrack.tk_eta(), sTrack.tk_phi()));

  reco::TrackBase::CovarianceMatrix cov;
  cov(0, 0) = pow(sTrack.tk_qoverp_Error(), 2);
  cov(0, 1) = sTrack.tk_qoverp_lambda_cov();
  cov(0, 2) = sTrack.tk_qoverp_phi_cov();
  cov(0, 3) = sTrack.tk_qoverp_dxy_cov();
  cov(0, 4) = sTrack.tk_qoverp_dsz_cov();
  cov(1, 1) = pow(sTrack.tk_lambda_Error(), 2);
  cov(1, 2) = sTrack.tk_lambda_phi_cov();
  cov(1, 3) = sTrack.tk_lambda_dxy_cov();
  cov(1, 4) = sTrack.tk_lambda_dsz_cov();
  cov(2, 2) = pow(sTrack.tk_phi_Error(), 2);
  cov(2, 3) = sTrack.tk_phi_dxy_cov();
  cov(2, 4) = sTrack.tk_phi_dsz_cov();
  cov(3, 3) = pow(sTrack.tk_dxy_Error(), 2);
  cov(3, 4) = sTrack.tk_dxy_dsz_cov();
  cov(4, 4) = pow(sTrack.tk_dsz_Error(), 2);

  return reco::Track(sTrack.tk_chi2(), sTrack.tk_ndof(), v, p, sTrack.tk_charge(), cov);
}

reco::Track pat::makeRecoTrack(const Run3ScoutingMuon& sMuon) {
  reco::Track::Point vtx(sMuon.trk_vx(), sMuon.trk_vy(), sMuon.trk_vz());
  reco::Track::Vector p3(math::RhoEtaPhiVector(sMuon.trk_pt(), sMuon.trk_eta(), sMuon.trk_phi()));

  reco::TrackBase::CovarianceMatrix cov;
  cov(0, 0) = pow(sMuon.trk_qoverpError(), 2);
  cov(0, 1) = sMuon.trk_qoverp_lambda_cov();
  cov(0, 2) = sMuon.trk_qoverp_phi_cov();
  cov(0, 3) = sMuon.trk_qoverp_dxy_cov();
  cov(0, 4) = sMuon.trk_qoverp_dsz_cov();
  cov(1, 1) = pow(sMuon.trk_lambdaError(), 2);
  cov(1, 2) = sMuon.trk_lambda_phi_cov();
  cov(1, 3) = sMuon.trk_lambda_dxy_cov();
  cov(1, 4) = sMuon.trk_lambda_dsz_cov();
  cov(2, 2) = pow(sMuon.trk_phiError(), 2);
  cov(2, 3) = sMuon.trk_phi_dxy_cov();
  cov(2, 4) = sMuon.trk_phi_dsz_cov();
  cov(3, 3) = pow(sMuon.trk_dxyError(), 2);
  cov(3, 4) = sMuon.trk_dxy_dsz_cov();
  cov(4, 4) = pow(sMuon.trk_dszError(), 2);

  return reco::Track(sMuon.trk_chi2(), sMuon.trk_ndof(), vtx, p3, sMuon.charge(), cov);
}

Run3ScoutingTrack pat::makeScoutingTrack(const reco::Track& trk) {
  return Run3ScoutingTrack(reduce_precision(trk.pt()),
                           reduce_precision(trk.eta()),
                           reduce_precision(trk.phi()),
                           reduce_precision(trk.chi2()),
                           trk.ndof(),
                           trk.charge(),
                           reduce_precision(trk.dxy()),
                           reduce_precision(trk.dz()),
                           trk.hitPattern().numberOfValidPixelHits(),
                           trk.hitPattern().trackerLayersWithMeasurement(),
                           trk.hitPattern().numberOfValidStripHits(),
                           reduce_precision(trk.qoverp()),
                           reduce_precision(trk.lambda()),
                           reduce_precision(trk.dxyError()),
                           reduce_precision(trk.dzError()),
                           reduce_precision(trk.qoverpError()),
                           reduce_precision(trk.lambdaError()),
                           reduce_precision(trk.phiError()),
                           reduce_precision(trk.dsz()),
                           reduce_precision(trk.dszError()),
                           reduce_precision(trk.covariance(0, 1)),
                           reduce_precision(trk.covariance(0, 2)),
                           reduce_precision(trk.covariance(0, 3)),
                           reduce_precision(trk.covariance(0, 4)),
                           reduce_precision(trk.covariance(1, 2)),
                           reduce_precision(trk.covariance(1, 3)),
                           reduce_precision(trk.covariance(1, 4)),
                           reduce_precision(trk.covariance(2, 3)),
                           reduce_precision(trk.covariance(2, 4)),
                           reduce_precision(trk.covariance(3, 4)),
                           0,
                           reduce_precision(trk.vx()),
                           reduce_precision(trk.vy()),
                           reduce_precision(trk.vz()));
}

pat::Muon pat::makePatMuon(const Run3ScoutingMuon& sMuon) {
  reco::Candidate::PolarLorentzVector p4(sMuon.pt(), sMuon.eta(), sMuon.phi(), sMuon.m());

  auto track = makeRecoTrack(sMuon);

  pat::Muon muon(reco::Muon(track.charge(), reco::Candidate::LorentzVector(p4), track.vertex()));

  muon.setType(sMuon.type());

  std::vector<reco::Track> tracks;
  tracks.push_back(track);
  muon.setGlobalTrack(reco::TrackRef(&tracks, 0));
  muon.embedCombinedMuon();
  muon.setBestTrack(reco::Muon::CombinedTrack);

  return muon;
}

reco::Vertex pat::makeRecoVertex(const Run3ScoutingVertex& sVertex) {
  reco::Vertex::Error err;
  err(0, 0) = pow(sVertex.xError(), 2);
  err(1, 1) = pow(sVertex.yError(), 2);
  err(2, 2) = pow(sVertex.zError(), 2);
  err(0, 1) = sVertex.xyCov();
  err(0, 2) = sVertex.xzCov();
  err(1, 2) = sVertex.yzCov();
  return reco::Vertex(reco::Vertex::Point(sVertex.x(), sVertex.y(), sVertex.z()),
                      err,
                      sVertex.chi2(),
                      sVertex.ndof(),
                      sVertex.tracksSize());
}

pat::PolarLorentzVector pat::makePolarLorentzVector(const Run3ScoutingTrack& trk, float mass) {
  return PolarLorentzVector(trk.tk_pt(), trk.tk_eta(), trk.tk_phi(), mass);
}
