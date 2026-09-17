#ifndef HLTriggerOffline_Scouting_ScoutingDataConversion_h
#define HLTriggerOffline_Scouting_ScoutingDataConversion_h

#include "DataFormats/Math/interface/Error.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/libminifloat.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/Scouting/interface/Run3ScoutingMuon.h"
#include "DataFormats/Scouting/interface/Run3ScoutingTrack.h"
#include "DataFormats/Scouting/interface/Run3ScoutingVertex.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

namespace dqm {
  
  inline reco::Track makeRecoTrack(const Run3ScoutingTrack& sTrack) {
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
  
  inline reco::Track makeRecoTrack(const Run3ScoutingMuon& sMuon) {
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
    
  inline reco::Vertex makeRecoVertex(const Run3ScoutingVertex& sVertex) {
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
}

#endif
