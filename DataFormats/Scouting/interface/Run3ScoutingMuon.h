#ifndef DataFormats_Run3ScoutingMuon_h
#define DataFormats_Run3ScoutingMuon_h

#include <vector>
#include "DataFormats/TrackReco/interface/Track.h"

// Class for holding muon information, for use in data scouting
// IMPORTANT: the content of this class should be changed only in backwards compatible ways!
class Run3ScoutingMuon {
public:
  //constructor with values for all data fields
  Run3ScoutingMuon(float pt,
                   float eta,
                   float phi,
                   float m,
                   unsigned int type,
                   int charge,
                   float normalizedChi2,
                   float ecalIso,
                   float hcalIso,
                   float trackIso,
                   int nValidStandAloneMuonHits,
                   int nStandAloneMuonMatchedStations,
                   int nValidRecoMuonHits,
                   int nRecoMuonChambers,
                   int nRecoMuonChambersCSCorDT,
                   int nRecoMuonMatches,
                   int nRecoMuonMatchedStations,
                   unsigned int nRecoMuonExpectedMatchedStations,
                   unsigned int recoMuonStationMask,
                   int nRecoMuonMatchedRPCLayers,
                   unsigned int recoMuonRPClayerMask,
                   int nValidPixelHits,
                   int nValidStripHits,
                   int nPixelLayersWithMeasurement,
                   int nTrackerLayersWithMeasurement,
                   float trk_chi2,
                   float trk_ndof,
                   float trk_dxy,
                   float trk_dz,
                   float trk_qoverp,
                   float trk_lambda,
                   float trk_pt,
                   float trk_phi,
                   float trk_eta,
                   float trk_dxyError,
                   float trk_dzError,
                   float trk_qoverpError,
                   float trk_lambdaError,
                   float trk_phiError,
                   float trk_dsz,
                   float trk_dszError,
                   float trk_qoverp_lambda_cov,
                   float trk_qoverp_phi_cov,
                   float trk_qoverp_dxy_cov,
                   float trk_qoverp_dsz_cov,
                   float trk_lambda_phi_cov,
                   float trk_lambda_dxy_cov,
                   float trk_lambda_dsz_cov,
                   float trk_phi_dxy_cov,
                   float trk_phi_dsz_cov,
                   float trk_dxy_dsz_cov,
                   float trk_vx,
                   float trk_vy,
                   float trk_vz,
                   reco::HitPattern trk_hitPattern,
                   std::vector<int> vtxIndx)
      : pt_(pt),
        eta_(eta),
        phi_(phi),
        m_(m),
        type_(type),
        charge_(charge),
        normalizedChi2_(normalizedChi2),
        ecalIso_(ecalIso),
        hcalIso_(hcalIso),
        trackIso_(trackIso),
        nValidStandAloneMuonHits_(nValidStandAloneMuonHits),
        nStandAloneMuonMatchedStations_(nStandAloneMuonMatchedStations),
        nValidRecoMuonHits_(nValidRecoMuonHits),
        nRecoMuonChambers_(nRecoMuonChambers),
        nRecoMuonChambersCSCorDT_(nRecoMuonChambersCSCorDT),
        nRecoMuonMatches_(nRecoMuonMatches),
        nRecoMuonMatchedStations_(nRecoMuonMatchedStations),
        nRecoMuonExpectedMatchedStations_(nRecoMuonExpectedMatchedStations),
        recoMuonStationMask_(recoMuonStationMask),
        nRecoMuonMatchedRPCLayers_(nRecoMuonMatchedRPCLayers),
        recoMuonRPClayerMask_(recoMuonRPClayerMask),
        nValidPixelHits_(nValidPixelHits),
        nValidStripHits_(nValidStripHits),
        nPixelLayersWithMeasurement_(nPixelLayersWithMeasurement),
        nTrackerLayersWithMeasurement_(nTrackerLayersWithMeasurement),
        trk_chi2_(trk_chi2),
        trk_ndof_(trk_ndof),
        trk_dxy_(trk_dxy),
        trk_dz_(trk_dz),
        trk_qoverp_(trk_qoverp),
        trk_lambda_(trk_lambda),
        trk_pt_(trk_pt),
        trk_phi_(trk_phi),
        trk_eta_(trk_eta),
        trk_dxyError_(trk_dxyError),
        trk_dzError_(trk_dzError),
        trk_qoverpError_(trk_qoverpError),
        trk_lambdaError_(trk_lambdaError),
        trk_phiError_(trk_phiError),
        trk_dsz_(trk_dsz),
        trk_dszError_(trk_dszError),
        trk_qoverp_lambda_cov_(trk_qoverp_lambda_cov),
        trk_qoverp_phi_cov_(trk_qoverp_phi_cov),
        trk_qoverp_dxy_cov_(trk_qoverp_dxy_cov),
        trk_qoverp_dsz_cov_(trk_qoverp_dsz_cov),
        trk_lambda_phi_cov_(trk_lambda_phi_cov),
        trk_lambda_dxy_cov_(trk_lambda_dxy_cov),
        trk_lambda_dsz_cov_(trk_lambda_dsz_cov),
        trk_phi_dxy_cov_(trk_phi_dxy_cov),
        trk_phi_dsz_cov_(trk_phi_dsz_cov),
        trk_dxy_dsz_cov_(trk_dxy_dsz_cov),
        trk_vx_(trk_vx),
        trk_vy_(trk_vy),
        trk_vz_(trk_vz),
        trk_hitPattern_(trk_hitPattern),
        vtxIndx_(std::move(vtxIndx)) {}
  //default constructor
  Run3ScoutingMuon()
      : pt_(0),
        eta_(0),
        phi_(0),
        m_(0),
        type_(0),
        charge_(0),
        normalizedChi2_(0),
        ecalIso_(0),
        hcalIso_(0),
        trackIso_(0),
        nValidStandAloneMuonHits_(0),
        nStandAloneMuonMatchedStations_(0),
        nValidRecoMuonHits_(0),
        nRecoMuonChambers_(0),
        nRecoMuonChambersCSCorDT_(0),
        nRecoMuonMatches_(0),
        nRecoMuonMatchedStations_(0),
        nRecoMuonExpectedMatchedStations_(0),
        recoMuonStationMask_(0),
        nRecoMuonMatchedRPCLayers_(0),
        recoMuonRPClayerMask_(0),
        nValidPixelHits_(0),
        nValidStripHits_(0),
        nPixelLayersWithMeasurement_(0),
        nTrackerLayersWithMeasurement_(0),
        trk_chi2_(0),
        trk_ndof_(0),
        trk_dxy_(0),
        trk_dz_(0),
        trk_qoverp_(0),
        trk_lambda_(0),
        trk_pt_(0),
        trk_phi_(0),
        trk_eta_(0),
        trk_dxyError_(0),
        trk_dzError_(0),
        trk_qoverpError_(0),
        trk_lambdaError_(0),
        trk_phiError_(0),
        trk_dsz_(0),
        trk_dszError_(0),
        trk_qoverp_lambda_cov_(0),
        trk_qoverp_phi_cov_(0),
        trk_qoverp_dxy_cov_(0),
        trk_qoverp_dsz_cov_(0),
        trk_lambda_phi_cov_(0),
        trk_lambda_dxy_cov_(0),
        trk_lambda_dsz_cov_(0),
        trk_phi_dxy_cov_(0),
        trk_phi_dsz_cov_(0),
        trk_dxy_dsz_cov_(0),
        trk_vx_(0),
        trk_vy_(0),
        trk_vz_(0),
        vtxIndx_(0) {}

  //accessor functions
  float pt() const { return pt_; }
  float eta() const { return eta_; }
  float phi() const { return phi_; }
  float m() const { return m_; }
  unsigned int type() const { return type_; }
  bool isGlobalMuon() const { return type_ & 1 << 1; }
  bool isTrackerMuon() const { return type_ & 1 << 2; }
  int charge() const { return charge_; }
  float normalizedChi2() const { return normalizedChi2_; }
  float ecalIso() const { return ecalIso_; }
  float hcalIso() const { return hcalIso_; }
  float trackIso() const { return trackIso_; }
  int nValidStandAloneMuonHits() const { return nValidStandAloneMuonHits_; }
  int nStandAloneMuonMatchedStations() const { return nStandAloneMuonMatchedStations_; }
  int nValidRecoMuonHits() const { return nValidRecoMuonHits_; }
  int nRecoMuonChambers() const { return nRecoMuonChambers_; }
  int nRecoMuonChambersCSCorDT() const { return nRecoMuonChambersCSCorDT_; }
  int nRecoMuonMatches() const { return nRecoMuonMatches_; }
  int nRecoMuonMatchedStations() const { return nRecoMuonMatchedStations_; }
  unsigned int nRecoMuonExpectedMatchedStations() const { return nRecoMuonExpectedMatchedStations_; }
  unsigned int recoMuonStationMask() const { return recoMuonStationMask_; }
  int nRecoMuonMatchedRPCLayers() const { return nRecoMuonMatchedRPCLayers_; }
  unsigned int recoMuonRPClayerMask() const { return recoMuonRPClayerMask_; }
  int nValidPixelHits() const { return nValidPixelHits_; }
  int nValidStripHits() const { return nValidStripHits_; }
  int nPixelLayersWithMeasurement() const { return nPixelLayersWithMeasurement_; }
  int nTrackerLayersWithMeasurement() const { return nTrackerLayersWithMeasurement_; }
  float trk_chi2() const { return trk_chi2_; }
  float trk_ndof() const { return trk_ndof_; }
  float trk_dxy() const { return trk_dxy_; }
  float trk_dz() const { return trk_dz_; }
  float trk_qoverp() const { return trk_qoverp_; }
  float trk_lambda() const { return trk_lambda_; }
  float trk_pt() const { return trk_pt_; }
  float trk_phi() const { return trk_phi_; }
  float trk_eta() const { return trk_eta_; }
  float trk_dxyError() const { return trk_dxyError_; }
  float trk_dzError() const { return trk_dzError_; }
  float trk_qoverpError() const { return trk_qoverpError_; }
  float trk_lambdaError() const { return trk_lambdaError_; }
  float trk_phiError() const { return trk_phiError_; }
  float trk_dsz() const { return trk_dsz_; }
  float trk_dszError() const { return trk_dszError_; }
  //add off-diagonal covariance matrix parameter, the above "Error" variables correspond to the diagonal, enum for Cov matrix (qoverp, lambda, phi, dxy, dsz), see https://github.com/cms-sw/cmssw/blob/CMSSW_11_2_X/DataFormats/TrackReco/src/TrackBase.cc for details
  float trk_qoverp_lambda_cov() const { return trk_qoverp_lambda_cov_; }
  float trk_qoverp_phi_cov() const { return trk_qoverp_phi_cov_; }
  float trk_qoverp_dxy_cov() const { return trk_qoverp_dxy_cov_; }
  float trk_qoverp_dsz_cov() const { return trk_qoverp_dsz_cov_; }
  float trk_lambda_phi_cov() const { return trk_lambda_phi_cov_; }
  float trk_lambda_dxy_cov() const { return trk_lambda_dxy_cov_; }
  float trk_lambda_dsz_cov() const { return trk_lambda_dsz_cov_; }
  float trk_phi_dxy_cov() const { return trk_phi_dxy_cov_; }
  float trk_phi_dsz_cov() const { return trk_phi_dsz_cov_; }
  float trk_dxy_dsz_cov() const { return trk_dxy_dsz_cov_; }
  float trk_vx() const { return trk_vx_; }
  float trk_vy() const { return trk_vy_; }
  float trk_vz() const { return trk_vz_; }
  reco::HitPattern const& trk_hitPattern() const { return trk_hitPattern_; }
  std::vector<int> const& vtxIndx() const { return vtxIndx_; }

private:
  float pt_;
  float eta_;
  float phi_;
  float m_;
  unsigned int type_;
  int charge_;
  float normalizedChi2_;
  float ecalIso_;
  float hcalIso_;
  float trackIso_;
  int nValidStandAloneMuonHits_;
  int nStandAloneMuonMatchedStations_;
  int nValidRecoMuonHits_;
  int nRecoMuonChambers_;
  int nRecoMuonChambersCSCorDT_;
  int nRecoMuonMatches_;
  int nRecoMuonMatchedStations_;
  unsigned int nRecoMuonExpectedMatchedStations_;
  unsigned int recoMuonStationMask_;
  int nRecoMuonMatchedRPCLayers_;
  unsigned int recoMuonRPClayerMask_;
  int nValidPixelHits_;
  int nValidStripHits_;
  int nPixelLayersWithMeasurement_;
  int nTrackerLayersWithMeasurement_;
  float trk_chi2_;
  float trk_ndof_;
  float trk_dxy_;
  float trk_dz_;
  float trk_qoverp_;
  float trk_lambda_;
  float trk_pt_;
  float trk_phi_;
  float trk_eta_;
  float trk_dxyError_;
  float trk_dzError_;
  float trk_qoverpError_;
  float trk_lambdaError_;
  float trk_phiError_;
  float trk_dsz_;
  float trk_dszError_;
  float trk_qoverp_lambda_cov_;
  float trk_qoverp_phi_cov_;
  float trk_qoverp_dxy_cov_;
  float trk_qoverp_dsz_cov_;
  float trk_lambda_phi_cov_;
  float trk_lambda_dxy_cov_;
  float trk_lambda_dsz_cov_;
  float trk_phi_dxy_cov_;
  float trk_phi_dsz_cov_;
  float trk_dxy_dsz_cov_;
  float trk_vx_;
  float trk_vy_;
  float trk_vz_;
  reco::HitPattern trk_hitPattern_;
  std::vector<int> vtxIndx_;
};

typedef std::vector<Run3ScoutingMuon> Run3ScoutingMuonCollection;

#endif
