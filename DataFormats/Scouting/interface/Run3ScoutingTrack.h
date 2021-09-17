#ifndef DataFormats_Run3ScoutingTrack_h
#define DataFormats_Run3ScoutingTrack_h

#include <vector>

//class for holding track information, for use in data scouting
// IMPORTANT: the content of this class should be changed only in backwards compatible ways!
class Run3ScoutingTrack {
public:
  //constructor with values for all data fields
  Run3ScoutingTrack(float tk_pt,
                    float tk_eta,
                    float tk_phi,
                    float tk_chi2,
                    float tk_ndof,
                    int tk_charge,
                    float tk_dxy,
                    float tk_dz,
                    int tk_nValidPixelHits,
                    int tk_nTrackerLayersWithMeasurement,
                    int tk_nValidStripHits,
                    float tk_qoverp,
                    float tk_lambda,
                    float tk_dxy_Error,
                    float tk_dz_Error,
                    float tk_qoverp_Error,
                    float tk_lambda_Error,
                    float tk_phi_Error,
                    float tk_dsz,
                    float tk_dsz_Error,
                    float tk_qoverp_lambda_cov,
                    float tk_qoverp_phi_cov,
                    float tk_qoverp_dxy_cov,
                    float tk_qoverp_dsz_cov,
                    float tk_lambda_phi_cov,
                    float tk_lambda_dxy_cov,
                    float tk_lambda_dsz_cov,
                    float tk_phi_dxy_cov,
                    float tk_phi_dsz_cov,
                    float tk_dxy_dsz_cov,
                    int tk_vtxInd,
                    float tk_vx,
                    float tk_vy,
                    float tk_vz)
      : tk_pt_(tk_pt),
        tk_eta_(tk_eta),
        tk_phi_(tk_phi),
        tk_chi2_(tk_chi2),
        tk_ndof_(tk_ndof),
        tk_charge_(tk_charge),
        tk_dxy_(tk_dxy),
        tk_dz_(tk_dz),
        tk_nValidPixelHits_(tk_nValidPixelHits),
        tk_nTrackerLayersWithMeasurement_(tk_nTrackerLayersWithMeasurement),
        tk_nValidStripHits_(tk_nValidStripHits),
        tk_qoverp_(tk_qoverp),
        tk_lambda_(tk_lambda),
        tk_dxy_Error_(tk_dxy_Error),
        tk_dz_Error_(tk_dz_Error),
        tk_qoverp_Error_(tk_qoverp_Error),
        tk_lambda_Error_(tk_lambda_Error),
        tk_phi_Error_(tk_phi_Error),
        tk_dsz_(tk_dsz),
        tk_dsz_Error_(tk_dsz_Error),
        tk_qoverp_lambda_cov_(tk_qoverp_lambda_cov),
        tk_qoverp_phi_cov_(tk_qoverp_phi_cov),
        tk_qoverp_dxy_cov_(tk_qoverp_dxy_cov),
        tk_qoverp_dsz_cov_(tk_qoverp_dsz_cov),
        tk_lambda_phi_cov_(tk_lambda_phi_cov),
        tk_lambda_dxy_cov_(tk_lambda_dxy_cov),
        tk_lambda_dsz_cov_(tk_lambda_dsz_cov),
        tk_phi_dxy_cov_(tk_phi_dxy_cov),
        tk_phi_dsz_cov_(tk_phi_dsz_cov),
        tk_dxy_dsz_cov_(tk_dxy_dsz_cov),
        tk_vtxInd_(tk_vtxInd),
        tk_vx_(tk_vx),
        tk_vy_(tk_vy),
        tk_vz_(tk_vz) {}
  //default constructor
  Run3ScoutingTrack()
      : tk_pt_(0),
        tk_eta_(0),
        tk_phi_(0),
        tk_chi2_(0),
        tk_ndof_(0),
        tk_charge_(0),
        tk_dxy_(0),
        tk_dz_(0),
        tk_nValidPixelHits_(0),
        tk_nTrackerLayersWithMeasurement_(0),
        tk_nValidStripHits_(0),
        tk_qoverp_(0),
        tk_lambda_(0),
        tk_dxy_Error_(0),
        tk_dz_Error_(0),
        tk_qoverp_Error_(0),
        tk_lambda_Error_(0),
        tk_phi_Error_(0),
        tk_dsz_(0),
        tk_dsz_Error_(0),
        tk_qoverp_lambda_cov_(0),
        tk_qoverp_phi_cov_(0),
        tk_qoverp_dxy_cov_(0),
        tk_qoverp_dsz_cov_(0),
        tk_lambda_phi_cov_(0),
        tk_lambda_dxy_cov_(0),
        tk_lambda_dsz_cov_(0),
        tk_phi_dxy_cov_(0),
        tk_phi_dsz_cov_(0),
        tk_dxy_dsz_cov_(0),
        tk_vtxInd_(0),
        tk_vx_(0),
        tk_vy_(0),
        tk_vz_(0) {}

  //accessor functions
  float tk_pt() const { return tk_pt_; }
  float tk_eta() const { return tk_eta_; }
  float tk_phi() const { return tk_phi_; }
  float tk_chi2() const { return tk_chi2_; }
  float tk_ndof() const { return tk_ndof_; }
  int tk_charge() const { return tk_charge_; }
  float tk_dxy() const { return tk_dxy_; }
  float tk_dz() const { return tk_dz_; }
  int tk_nValidPixelHits() const { return tk_nValidPixelHits_; }
  int tk_nTrackerLayersWithMeasurement() const { return tk_nTrackerLayersWithMeasurement_; }
  int tk_nValidStripHits() const { return tk_nValidStripHits_; }
  float tk_qoverp() const { return tk_qoverp_; }
  float tk_lambda() const { return tk_lambda_; }
  float tk_dxy_Error() const { return tk_dxy_Error_; }
  float tk_dz_Error() const { return tk_dz_Error_; }
  float tk_qoverp_Error() const { return tk_qoverp_Error_; }
  float tk_lambda_Error() const { return tk_lambda_Error_; }
  float tk_phi_Error() const { return tk_phi_Error_; }
  float tk_dsz() const { return tk_dsz_; }
  float tk_dsz_Error() const { return tk_dsz_Error_; }
  //add off-diagonal covariance matrix parameter, the above "Error" variables correspond to the diagonal, enum for Cov matrix (qoverp, lambda, phi, dxy, dsz), see https://github.com/cms-sw/cmssw/blob/CMSSW_11_2_X/DataFormats/TrackReco/src/TrackBase.cc for details
  float tk_qoverp_lambda_cov() const { return tk_qoverp_lambda_cov_; }
  float tk_qoverp_phi_cov() const { return tk_qoverp_phi_cov_; }
  float tk_qoverp_dxy_cov() const { return tk_qoverp_dxy_cov_; }
  float tk_qoverp_dsz_cov() const { return tk_qoverp_dsz_cov_; }
  float tk_lambda_phi_cov() const { return tk_lambda_phi_cov_; }
  float tk_lambda_dxy_cov() const { return tk_lambda_dxy_cov_; }
  float tk_lambda_dsz_cov() const { return tk_lambda_dsz_cov_; }
  float tk_phi_dxy_cov() const { return tk_phi_dxy_cov_; }
  float tk_phi_dsz_cov() const { return tk_phi_dsz_cov_; }
  float tk_dxy_dsz_cov() const { return tk_dxy_dsz_cov_; }
  int tk_vtxInd() const { return tk_vtxInd_; }
  float tk_vx() const { return tk_vx_; }
  float tk_vy() const { return tk_vy_; }
  float tk_vz() const { return tk_vz_; }

private:
  float tk_pt_;
  float tk_eta_;
  float tk_phi_;
  float tk_chi2_;
  float tk_ndof_;
  int tk_charge_;
  float tk_dxy_;
  float tk_dz_;
  int tk_nValidPixelHits_;
  int tk_nTrackerLayersWithMeasurement_;
  int tk_nValidStripHits_;
  float tk_qoverp_;
  float tk_lambda_;
  float tk_dxy_Error_;
  float tk_dz_Error_;
  float tk_qoverp_Error_;
  float tk_lambda_Error_;
  float tk_phi_Error_;
  float tk_dsz_;
  float tk_dsz_Error_;
  float tk_qoverp_lambda_cov_;
  float tk_qoverp_phi_cov_;
  float tk_qoverp_dxy_cov_;
  float tk_qoverp_dsz_cov_;
  float tk_lambda_phi_cov_;
  float tk_lambda_dxy_cov_;
  float tk_lambda_dsz_cov_;
  float tk_phi_dxy_cov_;
  float tk_phi_dsz_cov_;
  float tk_dxy_dsz_cov_;
  int tk_vtxInd_;
  float tk_vx_;
  float tk_vy_;
  float tk_vz_;
};

typedef std::vector<Run3ScoutingTrack> Run3ScoutingTrackCollection;

#endif
