///=== This is the Kalman Combinatorial Filter for 4 helix parameters track fit algorithm.
 
 
#include "L1Trigger/TrackFindingTMTT/interface/KFParamsComb.h"
#include "L1Trigger/TrackFindingTMTT/interface/KalmanState.h"
#include "L1Trigger/TrackFindingTMTT/interface/StubCluster.h"
#include "DataFormats/Math/interface/deltaPhi.h"
//#define CKF_DEBUG

namespace TMTT {


/*
// Scattering constants - HISTORIC NOT USED.

static unsigned nlayer_eta[25] = 
{ 6, 6, 6, 6,
6, 6, 6, 6, 6, 6, 6, 7, 7, 7,
7, 7, 7, 7, 6, 6, 6, 6, 6, 6};

static double matx_outer[25] = {
0.16, 0.17, 0.18, 0.19, 0.20, 
0.21, 0.26, 0.22, 0.26, 0.38,
0.41, 0.40, 0.44, 0.50, 0.54,
0.60, 0.44, 0.48, 0.60, 0.68,
0.50, 0.48, 0.64, 0.39, 0.20
};

static double matx_inner[25] = {
0.14, 0.1, 0.1, 0.1, 0.1, 
0.1, 0.1, 0.1, 0.1, 0.1, 
0.12, 0.1, 0.1, 0.1, 0.15,
0.20, 0.25, 0.25, 0.3, 0.3,
0.35, 0.40, 0.40, 0.6, 0.6
};
*/

KFParamsComb::KFParamsComb(const Settings* settings, const uint nPar, const string &fitterName ) : L1KalmanComb(settings, nPar, fitterName ){

  hdxmin[INV2R] = -1.1e-4;
  hdxmax[INV2R] = +1.1e-4;
  hdxmin[PHI0] = -6.e-3;
  hdxmax[PHI0] = +6.e-3;
  hdxmin[Z0] = -4.1;
  hdxmax[Z0] = +4.1;
  hdxmin[T] = -6.;
  hdxmax[T] = +6.;
  if (nPar_ == 5) {
    hdxmin[D0] = -1.001;
    hdxmax[D0] = +1.001;
  }

  hxmin[INV2R] = -0.3 * 0.0057;
  hxmax[INV2R] = +0.3 * 0.0057;
  hxmin[PHI0] = -0.3;
  hxmax[PHI0] = +0.3;
  hxmin[Z0] = -120;
  hxmax[Z0] = +120;
  hxmin[T] = -6.;
  hxmax[T] = +6.;
  if (nPar_ == 5) {
    hxmin[D0] = -3.5;
    hxmax[D0] = +3.5;
  }

  hddMeasmin[PHI0] = -1.e1;
  hddMeasmax[PHI0] = +1.e1;

  hresmin[PHI0] = -0.5;
  hresmax[PHI0] = +0.5;

  hresmin[PHI0] = -10.;
  hresmax[PHI0] = +10.;

  hxaxtmin[INV2R] = -1.e-3;
  hxaxtmax[INV2R] = +1.e-3;
  hxaxtmin[PHI0] = -1.e-1;
  hxaxtmax[PHI0] = +1.e-1;
  hxaxtmin[Z0] = -10.;
  hxaxtmax[Z0] = +10.;
  hxaxtmin[T] = -1.e-0;
  hxaxtmax[T] = +1.e-0;
  if (nPar_ == 5) {
    hxaxtmin[D0] = -1.001;
    hxaxtmax[D0] = +1.001;
  }
}


std::map<std::string, double> KFParamsComb::getTrackParams(const KalmanState *state )const{

  std::vector<double> x = state->xa();
  std::map<std::string, double> y;
  y["qOverPt"] = 2. * x.at(INV2R) / getSettings()->invPtToInvR(); 
  y["phi0"] = reco::deltaPhi( x.at(PHI0) + sectorPhi(), 0. );
  y["z0"] = x.at(Z0);
  y["t"] = x.at(T);
  if (nPar_ == 5) {
    y["d0"] = x.at(D0);
  }
  return y;
}

/* If using 5 param helix fit, get track params with beam-spot constraint & track fit chi2 from applying it. */
/* (N.B. chi2rz unchanged by constraint) */

std::map<std::string, double> KFParamsComb::getTrackParams_BeamConstr( const KalmanState *state, double& chi2rphi ) const {
  if (nPar_ == 5) {
    std::map<std::string, double> y;
    std::vector<double> x = state->xa();
    TMatrixD cov_xa       = state->pxxa(); 
    double deltaChi2rphi = (x.at(D0) * x.at(D0)) / cov_xa[D0][D0];
    chi2rphi = state->chi2rphi() + deltaChi2rphi;
    // Apply beam-spot constraint to helix params in transverse plane only, as most sensitive to it.
    x[INV2R] -= x.at(D0) * (cov_xa[INV2R][D0] / cov_xa[D0][D0]); 
    x[PHI0 ] -= x.at(D0) * (cov_xa[PHI0 ][D0] / cov_xa[D0][D0]); 
    x[D0   ]  = 0.0;
    y["qOverPt"] = 2. * x.at(INV2R) / getSettings()->invPtToInvR(); 
    y["phi0"]    = reco::deltaPhi( x.at(PHI0) + sectorPhi(), 0. );
    y["z0"]      = x.at(Z0);
    y["t"]       = x.at(T);
    y["d0"]      = x.at(D0);
    return y;
  } else {
    return (this->getTrackParams(state));
  }
}

 
/* The Kalman measurement matrix = derivative of helix intercept w.r.t. helix params 
 * Here I always measure phi(r), and z(r) */
TMatrixD KFParamsComb::H(const StubCluster* stubCluster)const{
  TMatrixD h(2, nPar_);
  double r = stubCluster->r();
  h(PHI,INV2R) = -r;
  h(PHI,PHI0) = 1;
  if (nPar_ == 5) {
    h(PHI,D0) = -1./r;
  }
  h(Z,Z0) = 1;
  h(Z,T) = r;
  return h;
}

// Not used?

TMatrixD KFParamsComb::dH(const StubCluster* stubCluster)const{

  double dr(0);
  if(stubCluster->layerId() > 10){
    dr = stubCluster->sigmaZ();
  }

  double r = stubCluster->r();

  TMatrixD h(2, nPar_);
  h(PHI,INV2R) = -dr;
  if (nPar_ == 5) {
    h(PHI,D0) = dr/(r*r);
  }
  h(Z,T) = dr;

  return h;
}
 
/* Seed the state vector */
std::vector<double> KFParamsComb::seedx(const L1track3D& l1track3D)const{

  std::vector<double> x(nPar_);
  x[INV2R] = getSettings()->invPtToInvR() * l1track3D.qOverPt()/2;
  x[PHI0]  = reco::deltaPhi( l1track3D.phi0() - sectorPhi(), 0. );
  x[Z0]    = l1track3D.z0();
  x[T]     = l1track3D.tanLambda();
  if (nPar_ == 5) {
    x[D0]    = l1track3D.d0();
  }
    
  return x;
}

/* Seed the covariance matrix */
TMatrixD KFParamsComb::seedP(const L1track3D& l1track3D)const{
  TMatrixD p(nPar_,nPar_);

  double invPtToInv2R = getSettings()->invPtToInvR() / 2; 

  // Assumed track seed (from HT) uncertainty in transverse impact parameter.
  const float d0Sigma = 1.0;

  if (getSettings()->hybrid()) {

    p(INV2R,INV2R) = 0.0157 * 0.0157 * invPtToInv2R * invPtToInv2R * 4; 
    p(PHI0,PHI0) = 0.0051 * 0.0051 * 4; 
    p(Z0,Z0) = 5.0 * 5.0; 
    p(T,T) = 0.25 * 0.25 * 4;
    // N.B. (z0, tanL, d0) seed uncertainties could be smaller for hybrid, if seeded in PS? -- not tried
    //if (l1track3D.seedPS() > 0) { // Tracklet seed used PS layers
    //  p(Z0,Z0) /= (4.*4.).;
    //  p(T,T) /= (4.*4.);
    // }
    if (nPar_ == 5) {
      p(D0,D0) = d0Sigma * d0Sigma; 
    } 

  } else {

    // optimised for 18x2 with additional error factor in pt/phi to avoid pulling towards wrong HT params
    p(INV2R,INV2R) = 0.0157 * 0.0157 * invPtToInv2R * invPtToInv2R * 4;  // Base on HT cell size
    p(PHI0,PHI0) = 0.0051 * 0.0051 * 4; // Based on HT cell size.
    p(Z0,Z0) = 5.0 * 5.0; 
    p(T,T) = 0.25 * 0.25 * 4; // IRT: increased by factor 4, as was affecting fit chi2.
    if (nPar_ == 5) {
      p(D0,D0) = d0Sigma * d0Sigma; 
    } 

    if ( getSettings()->numEtaRegions() <= 12 ) {    
      // Inflate eta errors
      p(T,T) = p(T,T) * 2 * 2;
    }
  }

  return p;
}

/* The forecast matrix
 * (here equals identity matrix) */
TMatrixD KFParamsComb::F(const StubCluster* stubCluster, const KalmanState *state )const{
  TMatrixD F(nPar_,nPar_); 
  for(unsigned int n = 0; n < nPar_; n++)
    F(n, n) = 1;
  return F;
}

/* the vector of measurements */
std::vector<double> KFParamsComb::d(const StubCluster* stubCluster )const{
  std::vector<double> meas;
  meas.resize(2);
  meas[PHI] = reco::deltaPhi( stubCluster->phi(), sectorPhi() );
  meas[Z] = stubCluster->z();
  return meas;
}

// Assumed hit resolution in (phi,z)
TMatrixD KFParamsComb::PddMeas(const StubCluster* stubCluster, const KalmanState *state )const{

  double inv2R = (getSettings()->invPtToInvR()) * 0.5 * state->candidate().qOverPt(); // alternatively use state->xa().at(INV2R)
  double inv2R2 = inv2R * inv2R;

  double tanl = state->xa().at(T);  // factor of 0.9 improves rejection
  double tanl2 = tanl * tanl; 

  TMatrixD p(2,2);

  double vphi(0);
  double vz(0);
  double vcorr(0);

  // consider error due to integerisation only for z (r in encap) coord when enabled
  double err_digi2(0);
  if (getSettings()->enableDigitize()) err_digi2 = 0.15625 * 0.15625 / 12.0;

  double a = stubCluster->sigmaX() * stubCluster->sigmaX();
  double b = stubCluster->sigmaZ() * stubCluster->sigmaZ() + err_digi2;
  double r2 = stubCluster->r() * stubCluster->r();
  double invr2 = 1./r2;

  // Scattering term scaling as 1/Pt.
  double sigmaScat = getSettings()->kalmanMultiScattTerm()/(state->candidate().pt());
  double sigmaScat2 = sigmaScat * sigmaScat;

  if ( stubCluster->barrel() ) {

    vphi = (a * invr2) + sigmaScat2;

    if (stubCluster->tiltedBarrel()) {
      // Convert uncertainty in (r,phi) to (z,phi).
      float scaleTilted = 1.;
      if (getSettings()->kalmanHOtilted()) {
	if ( getSettings()->useApproxB() ) { // Simple firmware approximation
	  scaleTilted = getApproxB(stubCluster->z(), stubCluster->r());
	} else {                             // Exact C++ implementation. 
	  float tilt = stubCluster->moduleTilt();
	  scaleTilted = sin(tilt) + cos(tilt)*tanl;
	}
      }
      float scaleTilted2 = scaleTilted*scaleTilted;
      // This neglects the non-radial strip effect, assumed negligeable for PS.
      vz = b * scaleTilted2;
    } else {
      vz = b;
    }

    if (getSettings()->kalmanHOdodgy()) {
      // Use original (Dec. 2016) dodgy implementation was this.
      vz = b;
    }

  } else {

    vphi = a * invr2 + sigmaScat2;
    vz = (b * tanl2);

    if (not stubCluster->psModule()) {   // Neglect these terms in PS 
      double beta = 0.;
      // Add correlation term related to conversion of stub residuals from (r,phi) to (z,phi).
      if (getSettings()->kalmanHOprojZcorr() == 2) beta += -inv2R;
      // Add alpha correction for non-radial 2S endcap strips..
      if (getSettings()->kalmanHOalpha()     == 2) beta += -stubCluster->alpha();  // alpha is 0 except in endcap 2S disks

      double beta2 = beta * beta;
      vphi += b * beta2;
      vcorr = b * (beta * tanl);

      // IRT - for checking efficiency of removing phi-z correlation from projection.
      // "ultimate_off1"
      //vphi  = a * invr2 + b * pow(-stubCluster->alpha(), 2) + b * inv2R2 + sigmaScat2;
      //vcorr = b * ((-stubCluster->alpha()) * tanl);

      // IRT - This higher order correction doesn't significantly improve the track fit performance, so commented out.
      //if (getSettings()->kalmanHOhelixExp()) {
      //  float dsByDr = 1. + (1./2.)*r2*inv2R2; // Allows for z = z0 + s*tanL, where s is not exactly r due to circle.
      //  vcorr *= dsByDr;
      //  vz *= dsByDr * dsByDr;
      //}

      if (getSettings()->kalmanHOdodgy()) {
        // Use original (Dec. 2016) dodgy implementation was this.
        vphi = (a * invr2) + (b * inv2R2) + sigmaScat2;
        vcorr = 0.;
	vz = (b * tanl2);
      }
    }
  }

  p(PHI, PHI) = vphi;
  p(Z, Z) = vz;
  p(PHI, Z) = vcorr;
  p(Z, PHI) = vcorr;

  return p;

}

// State uncertainty due to scattering -- HISTORIC NOT USED
TMatrixD KFParamsComb::PxxModel( const KalmanState *state, const StubCluster *stubCluster )const
{

  TMatrixD p(nPar_,nPar_);

  /*
    if( getSettings()->kalmanMultiScattFactor() ){

    unsigned i_eta = abs( stubCluster->eta() / 0.1 );
    if( i_eta > 24 ) i_eta = 24;
    double dl = matx_outer[i_eta] / nlayer_eta[i_eta];

    unsigned stub_itr = state->nextLayer();

    const KalmanState * last_update_state = state->last_update_state();
    unsigned last_itr(1);
    if( last_update_state ) last_itr = last_update_state->nextLayer();
    dl = ( stub_itr - last_itr ) * dl; 

    if( dl ){
    std::map<std::string, double> y = getTrackParams( state );
    double dtheta0 = 1./sqrt(3) * 0.0136 * fabs(y["qOverPt"]) * sqrt(dl)*( 1+0.038*log(dl) ); 
    dtheta0 *= getSettings()->kalmanMultiScattFactor();
    p(PHI0, PHI0) = dtheta0 * dtheta0; // Despite the name, I think this is uncertainty in phi0. I guess uncertainty in theta0 neglected compared to detector resolution.
    }
    }
  */

  return p;
}

bool KFParamsComb::isGoodState( const KalmanState &state )const
{
  // Cut values. (Layer 0 entry here is dummy). -- todo : make configurable

  vector<float> z0Cut, ptTolerance, d0Cut, chi2Cut;
  //  Layer   =    0      1      2     3     4      5      6
  ptTolerance = { 999.,  999.,   0.1,  0.1,  0.05, 0.05,  0.05};
  d0Cut       = { 999.,  999.,     999.,      10.,      10.,      10.,       10.}; // Only used for 5 param fit.
  if (nPar_ == 5) { // specific cuts for displaced tracking case.
    //  Layer   =    0      1        2         3         4         5           6
    z0Cut       = { 999.,  999.,  1.7*15.,  1.7*15.,  1.7*15.,  1.7*15.,   1.7*15.}; // Larger values require digisation change.
    chi2Cut     = { 999.,  999.,      10.,      30.,      80.,     120.,      160.}; // Maybe loosen for high d0 ?
  } else {         // specific cuts for prompt tracking case.
    //  Layer   =    0      1      2     3     4      5      6
    z0Cut       = { 999.,  999.,   15.,  15.,  15.,   15.,   15.};
    chi2Cut     = { 999.,  999.,   10.,  30.,  80.,  120.,  160.};
  }
  
  unsigned nStubLayers = state.nStubLayers();
  bool goodState( true );

  std::map<std::string, double> y = getTrackParams( &state );
  double qOverPt = y["qOverPt"]; 
  double pt=fabs( 1/qOverPt ); 
  double z0=fabs( y["z0"] ); 

  // state parameter selections

  if (z0 > z0Cut[nStubLayers] ) goodState = false;
  if( pt < getSettings()->houghMinPt() - ptTolerance[nStubLayers] ) goodState = false;
  if (nPar_ == 5) {
    double d0=fabs( state.xa()[D0] ); 
    if( d0 > d0Cut[nStubLayers] ) goodState = false;
  }

  // chi2 selection

  double chi2scaled = state.chi2scaled(); // chi2(r-phi) scaled down to improve electron performance.

  if (getSettings()->kalmanMultiScattTerm() > 0.0001) {   // Scattering taken into account

    if (chi2scaled > chi2Cut[nStubLayers]) goodState=false; // No separate pT selection needed

  } else {  // scattering ignored - HISTORIC

    // N.B. Code below changed by Alexander Morton to allow tracking down to Pt = 2 GeV.
    if( nStubLayers == 2 ) {
      if (chi2scaled > 15.0) goodState=false; // No separate pT selection needed
    } else if ( nStubLayers == 3 ) {
      if (chi2scaled > 100.0 && pt > 2.7) goodState=false;
      if (chi2scaled > 120.0 && pt <= 2.7) goodState=false;
    } else if ( nStubLayers == 4 ) { 
      if (chi2scaled > 320.0 && pt > 2.7) goodState=false;
      if (chi2scaled > 1420.0 && pt <= 2.7) goodState=false;
    } else if ( nStubLayers == 5 ) {  // NEEDS TUNING FOR 5 OR 6 LAYERS !!!
      if (chi2scaled > 480.0 && pt > 2.7) goodState=false;
      if (chi2scaled > 2130.0 && pt <= 2.7) goodState=false;
    } else if ( nStubLayers >= 6 ) {  // NEEDS TUNING FOR 5 OR 6 LAYERS !!!
      if (chi2scaled > 640.0 && pt > 2.7) goodState=false;
      if (chi2scaled > 2840.0 && pt <= 2.7) goodState=false;
    }

  }

  const bool countUpdateCalls = false; // Print statement to count calls to Updator.

  if ( countUpdateCalls || 
       (getSettings()->kalmanDebugLevel() >= 2 && tpa_ != nullptr) ||
       (getSettings()->kalmanDebugLevel() >= 2 && getSettings()->hybrid()) ) {
    if (not goodState) cout<<"State veto:";
    if (goodState)     cout<<"State kept:"; 
    cout<<" nlay="<<nStubLayers<<" nskip="<<state.nSkippedLayers()<<" chi2_scaled="<<chi2scaled;
    if (tpa_ != nullptr) cout<<" pt(mc)="<<tpa_->pt();
    cout<<" pt="<<pt<<" q/pt="<<qOverPt<<" tanL="<<y["t"]<<" z0="<<y["z0"]<<" phi0="<<y["phi0"];
    if (nPar_ == 5) cout<<" d0="<<y["d0"];
    cout<<" fake"<<(tpa_ == nullptr);
    if (tpa_ != nullptr) cout<<" pt(mc)="<<tpa_->pt();
    cout<<endl;
  }

  return goodState;
}

}

