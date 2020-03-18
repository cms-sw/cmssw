/**
 * This is the interface between the C++ KF framework in CMSSW and the HLS code.
 * It is identical to KFParamsComb, except that the state updator is modified 
 *
 * All variable names & equations come from Fruhwirth KF paper
 * http://dx.doi.org/10.1016/0168-9002%2887%2990887-4
 * 
 * Author: Ian Tomalin
 */

#include "L1Trigger/TrackFindingTMTT/interface/HLS/KFParamsCombCallHLS.h"
#include "L1Trigger/TrackFindingTMTT/interface/HLS/KFconstants.h"
#include "L1Trigger/TrackFindingTMTT/interface/HLS/KalmanUpdate.h"

#include "L1Trigger/TrackFindingTMTT/interface/TP.h"
#include "L1Trigger/TrackFindingTMTT/interface/StubCluster.h"
#include "L1Trigger/TrackFindingTMTT/interface/KalmanState.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <vector>
#include <iostream>

namespace TMTT {

//--- Explicit instantiation required for all non-specialized templates, to allow them to be implemented 
//--- in .cc files.

template KalmanHLS::KFstate<4> KFParamsCombCallHLS::getDigiStateIn(unsigned int skipped, unsigned int layer, const KalmanState* state) const;

template KalmanHLS::KFstate<5> KFParamsCombCallHLS::getDigiStateIn(unsigned int skipped, unsigned int layer, const KalmanState* state) const;

template const KalmanState* KFParamsCombCallHLS::getStateOut<4>(const KalmanState* stateIn, const StubCluster* stubCluster, const KalmanHLS::KFstate<4>& stateOutDigi, const KalmanHLS::KFselect<4>& selectOutDigi);

template const KalmanState* KFParamsCombCallHLS::getStateOut<5>(const KalmanState* stateIn, const StubCluster* stubCluster, const KalmanHLS::KFstate<5>& stateOutDigi, const KalmanHLS::KFselect<5>& selectOutDigi);

//--- Normal code below ...

KFParamsCombCallHLS::KFParamsCombCallHLS(const Settings* settings, const uint nPar, const string &fitterName ) : KFParamsComb(settings, nPar, fitterName) {

  // Get digitisation multipliers.
  rMult_   = pow(2, getSettings()->rtBits() )   / (getSettings()->rtRange());
  zMult_   = pow(2, getSettings()->zBits() )    / (getSettings()->zRange()); 
  phiMult_ = pow(2, getSettings()->phiSBits() ) / (getSettings()->phiSRange()); 
  const double small = 1.e-6;
  if (fabs(rMult_ - KalmanHLS::rMult) / rMult_ > small || fabs(zMult_ - KalmanHLS::rMult / 2) / zMult_ > small || fabs(phiMult_ - KalmanHLS::phiMult) / phiMult_ > small) {
	throw cms::Exception("ERROR: KFParamsCombCallHLS inconsistent digi multipliers.")<<" r="<<rMult_<<" "<<KalmanHLS::rMult<<"  z="<<zMult_<<" "<<(KalmanHLS::rMult / 2)<<" phi="<<phiMult_<<" "<<KalmanHLS::phiMult<<std::endl; 
  }
  // Multiplier of (phiMult/rMult) for helix param "inv2R" simplifies the KF maths, as explained in
  // https://svnweb.cern.ch/cern/wsvn/UK-TrackTrig/demonstrator/specifications/demonstrator2_formats_working_doc.docx 
  inv2R_Mult_ = (phiMult_/rMult_);
  d0_Mult_ = (phiMult_*rMult_);

  // Reference radius in r-phi plane.
  chosenRofPhi_ = getSettings()->chosenRofPhi();
  // Number of eta sectors.
  numEtaRegions_ = getSettings()->numEtaRegions();

#ifdef PT_2GEV
  if (settings->houghMinPt() > 2.5) throw cms::Exception("KFParamsConmbCallHLS: Edit KFpragmaOpts.h to undefine PT_2GEV");
#else
  if (settings->houghMinPt() < 2.5) throw cms::Exception("KFParamsConmbCallHLS: Edit KFpragmaOpts.h to define PT_2GEV");
#endif

#ifdef HYBRID_FORMAT
  if (not settings_->hybrid()) throw cms::Exception("KFParamsConmbCallHLS: Edit KFpragmaOpts.h to undefine HYBRID_FORMAT");
#else
  if (settings_->hybrid()) throw cms::Exception("KFParamsConmbCallHLS: Edit KFpragmaOpts.h to define HYBRID_FORMAT");
#endif
}

//=== Update KF helix params with this stub.
//=== (Override KF state updator in L1KalmanComb with version suitable for HLS).

const KalmanState* KFParamsCombCallHLS::kalmanUpdate( unsigned skipped, unsigned layer, const StubCluster *stubCluster, const KalmanState &stateIn, const TP *tpa ) {

  //  cout.setf(ios::scientific, ios::floatfield); // Get useful debug printout ...
  cout.unsetf(ios::floatfield); // Get useful debug printout ...
  cout.precision(8);
  
  // Get digitised stub info
  KalmanHLS::KFstubC stubDigi = this->getDigiStub(stubCluster, &stateIn);

  if (nPar_ == 4) {

    // Get digitised KF state info
    KalmanHLS::KFstate<4> stateInDigi = this->getDigiStateIn<4>(skipped, layer, &stateIn);

    // Call HLS code to add stub to helix state.
    KalmanHLS::KFstate<4> stateOutDigi;
    KalmanHLS::KFselect<4> selectOutDigi;
    KalmanHLS::kalmanUpdate(stubDigi, stateInDigi, stateOutDigi, selectOutDigi);

    // Convert digitized ourput KF state to floating point.
    const KalmanState* newState = this->getStateOut(&stateIn, stubCluster, stateOutDigi, selectOutDigi);

    return newState;

  } else {

    // Get digitised KF state info
    KalmanHLS::KFstate<5> stateInDigi = this->getDigiStateIn<5>(skipped, layer, &stateIn);

    // Call HLS code to add stub to helix state.
    KalmanHLS::KFstate<5> stateOutDigi;
    KalmanHLS::KFselect<5> selectOutDigi;
    KalmanHLS::kalmanUpdate(stubDigi, stateInDigi, stateOutDigi, selectOutDigi);

    // Convert digitized ourput KF state to floating point.
    const KalmanState* newState = this->getStateOut(&stateIn, stubCluster, stateOutDigi, selectOutDigi);

    return newState;
  }
}

//=== Get digital stub that the KF VHDL injects into the KF state updater (Maxeller/HLS)

KalmanHLS::KFstubC KFParamsCombCallHLS::getDigiStub(const StubCluster* stubCluster, const KalmanState* state) {
  // Get digitised stub(s) making up stub cluster.
  const vector<const Stub*> stubs = stubCluster->stubs();
  if (stubs.size() != 1) throw cms::Exception("KFParamsCombCallHLS: Can't cope with StubCluster that doesn't contain a single stub")<<stubs.size()<<endl;

  const DigitalStub& digiStub = stubs[0]->digitalStub();
 
  KalmanHLS::KFstubC stubDigi;
  // KF uses stub r, not rT. 
  stubDigi.r    = digiStub.iDigi_Rt() + std::round(rMult_*chosenRofPhi_); 
  // KF VHDL z multiplier is factor 2 smaller than r multiplier, with BZ = BZ1 - 1
  // bits assigned to integer part of z in VHDL, and BZ1 bits assigned to it in HLS. Following statement
  // therefore doubles the effective z multiplier.
  enum {BZ = KalmanHLS::KFstubN::BZ, BZ1 = KalmanHLS::KFstubN::BZ1};
  stubDigi.z.range(BZ-1, 0) = ap_fixed<BZ,BZ>(digiStub.iDigi_Z_KF()).range(BZ-1, 0); 
  stubDigi.phiS = digiStub.iDigi_PhiS();

  #ifdef IRT_DEBUG
  if (state->candidate().getMatchedTP() != nullptr) {
    unsigned int iPhiSec = state->candidate().iPhiSec();

    // Centre of phi (tracking) nonant zero must be along x-axis to be consistent with tracker cabling map.
    // Define phi sector zero  to start at lower end of phi range in nonant 0.
    float phiCentreSec0 = -M_PI/float(getSettings()->numPhiNonants()) + M_PI/float(getSettings()->numPhiSectors());
    float phiSec = 2.*M_PI * float(iPhiSec) / float(getSettings()->numPhiSectors()) + phiCentreSec0; // Centre of sector in phi

    float phiStubOff = reco::deltaPhi(digiStub.phi(), digiStub.phiS());
    cout<<"KF sector phi check "<<phiSec<<" "<<phiStubOff<<endl;
    cout<<"KF input stub: float (r,phi) = "<<stubs[0]->r()<<" "<<reco::deltaPhi(stubs[0]->phi(),phiSec)<<endl;
    cout<<"KF input stub: digi (r,phi) = "<<double(stubDigi.r)/rMult_<<" "<<double(stubDigi.phiS)/phiMult_<<endl;
  }
  #endif

#ifdef IRT_DEBUG
  cout<<"PS MODULE "<<stubs[0]->psModule()<<endl;;
#endif

  stubDigi.valid = true;

  return stubDigi;
}

//=== Get digitised KF state info that the KF VHDL injects into the KF state updater (Maxeller/HLS)
//=== for NPAR = 4 & 5 param helix fits.

template <unsigned int NPAR>
KalmanHLS::KFstate<NPAR> KFParamsCombCallHLS::getDigiStateIn(unsigned int skipped, unsigned int layer, const KalmanState* state) const {
  // Calculate factors to convert floating point helix params to digitized ones.
  // Based on constants & functions named *HWU* in
//https://svnweb.cern.ch/cern/wsvn/UK-TrackTrig/firmware/trunk/cactusupgrades/projects/tracktrigger/kalmanfit/firmware/cgn/src/formats/Constants.maxj .

  KalmanHLS::KFstate<NPAR> stateDigi;

  // Cell HT found track in, with (mbin,cbin) centred on zero.
  pair<unsigned int, unsigned int> htCell = state->candidate().getCellLocationHT();
  // Digitized HT cell info must be centred on zero. (See L1fittedTrack::digitalTrack()).
  stateDigi.mBin_ht = htCell.first   - floor(getSettings()->houghNbinsPt()/2); 
  stateDigi.cBin_ht = htCell.second  - floor(getSettings()->houghNbinsPhi()/2);

  stateDigi.hitPattern = state->hitPattern();

  // Fitted helix parameters.
  vector<double> helixParams = state->xa();
  double inv2R = helixParams[0]; // Half inverse radius of curvature.
  double phi0  = helixParams[1]; // Measured with respect to centre of phi sector.
  double tanL  = helixParams[2];
  double z0    = helixParams[3];
  TMatrixD cov = state->pxxa();
  double cov_inv2R_inv2R = cov[0][0];
  double cov_phi0_phi0   = cov[1][1];
  double cov_tanL_tanL   = cov[2][2];
  double cov_z0_z0       = cov[3][3];
  double cov_inv2R_phi0  = cov[0][1];
  double cov_tanL_z0     = cov[2][3];  

  // Digitize helix parameters, using multiplication factors in 
  // Demonstrator Units" section of https://twiki.cern.ch/twiki/pub/CMS/CMSUKTrkTrig/KFequations.pdf .

  // Using the specified multipliers, which are related to those used for stubs (see getDigiStub()) simplifies the maths.
  // The helix params need better granularity than the stubs, which is accomodated by using additional bits after the 
  // decimal point in the digitized numbers (profitting from Maxeller/HLS ability to handle floats).

  stateDigi.inv2R = inv2R*inv2R_Mult_; // See inv2RToHWU() in Maxeller code in above web link.
  stateDigi.phi0  = phi0*phiMult_;

  #ifdef IRT_DEBUG
  if (state->candidate().getMatchedTP() != nullptr) {
    unsigned int iPhiSec = state->candidate().iPhiSec();
    float phiCentreSec0 = -M_PI/float(getSettings()->numPhiNonants()) + M_PI/float(getSettings()->numPhiSectors());
    float phiSec = 2.*M_PI * float(iPhiSec) / float(getSettings()->numPhiSectors()) + phiCentreSec0; // Centre of sector in phi
    cout<<"KF Input track (float): q/pt = "<<inv2R/(0.5*getSettings()->invPtToInvR())<<" phi0 = "<<phi0<<endl;
    cout<<"KF Input track (digi): q/pt = "<<double(stateDigi.inv2R)/inv2R_Mult_/(0.5*getSettings()->invPtToInvR())<<" phi0 = "<<double(stateDigi.phi0)/phiMult_<<endl;
    cout<<"          truth: q/pt = "<<state->candidate().getMatchedTP()->qOverPt()<<" phi0 = "<<reco::deltaPhi(state->candidate().getMatchedTP()->phi0(),phiSec)<<endl;
  }
  #endif

  stateDigi.tanL  = tanL;              // Multiplier is 1 for tanL.
  // Multiplier of rMult instead of zMult here simplifies maths, like for stub z in getDigiStub().
  stateDigi.z0    = z0*rMult_;         // See rMmToHWU() in Maxeller code.

  // Check digitisation range of helix parameters is sufficient.
  KalmanHLS::CHECK_AP::checkCalc("helix0", stateDigi.inv2R, inv2R*inv2R_Mult_, 9.9e9, 0.0001);
  KalmanHLS::CHECK_AP::checkCalc("helix1", stateDigi.phi0 , phi0*phiMult_    , 9.9e9, 0.001);
  KalmanHLS::CHECK_AP::checkCalc("helix2", stateDigi.tanL , tanL             , 9.9e9, 0.001);
  KalmanHLS::CHECK_AP::checkCalc("helix3", stateDigi.z0   , z0*rMult_        , 9.9e9, 0.1);

  // Multipliers for covariance matrix follow from those of helix params.
  stateDigi.cov_00 = cov_inv2R_inv2R * inv2R_Mult_ * inv2R_Mult_;
  stateDigi.cov_11 = cov_phi0_phi0 * phiMult_ * phiMult_;
  stateDigi.cov_22 = cov_tanL_tanL;
  stateDigi.cov_33 = cov_z0_z0 * rMult_ * rMult_;
  stateDigi.cov_01 = cov_inv2R_phi0 * inv2R_Mult_ * phiMult_;
  stateDigi.cov_23 = cov_tanL_z0 * rMult_;

  // Check digitisation range of covariance matrix is sufficient.
  KalmanHLS::CHECK_AP::checkCalc("C00_old", stateDigi.cov_00, cov_inv2R_inv2R * inv2R_Mult_ * inv2R_Mult_);
  KalmanHLS::CHECK_AP::checkCalc("C11_old", stateDigi.cov_11, cov_phi0_phi0   * phiMult_    * phiMult_);
  KalmanHLS::CHECK_AP::checkCalc("C22_old", stateDigi.cov_22, cov_tanL_tanL);
  KalmanHLS::CHECK_AP::checkCalc("C33_old", stateDigi.cov_33, cov_z0_z0       * rMult_      * rMult_);
  KalmanHLS::CHECK_AP::checkCalc("C01_old", stateDigi.cov_01, cov_inv2R_phi0  * inv2R_Mult_ * phiMult_);
  KalmanHLS::CHECK_AP::checkCalc("C23_old", stateDigi.cov_23, cov_tanL_z0     * rMult_);

  this->getDigiStateInUtil(helixParams, cov, stateDigi);

  stateDigi.chiSquaredRphi = state->chi2rphi();
  stateDigi.chiSquaredRz   = state->chi2rz();

  // This is the KF layer that we are currently looking for stubs in, incremented by L1KalmanComb::doKF(), which in any eta region increases from 0-7 as a particle goes through each layer in turn.
  stateDigi.layerID        = layer;
  // This is the number of skipped layers assuming we find a stub in the layer currently being searched.
  stateDigi.nSkippedLayers = skipped;

  stateDigi.trackID = 0; // Not used by KF updator.
  stateDigi.eventID = 0; // Not used by KF updator.

  unsigned int iEtaReg = state->candidate().iEtaReg(); // Although this comes from the state, it is actually the eta region of the stub.
  // This is encoded in tortuous way copied from Maxeller code (lines 127-133).
//https://svnweb.cern.ch/cern/wsvn/UK-TrackTrig/firmware/trunk/cactusupgrades/projects/tracktrigger/kalmanfit/firmware/hdl/packages/stubs.vhd
  if (iEtaReg < numEtaRegions_/2) {
    stateDigi.etaSectID = (numEtaRegions_/2 - 1) - iEtaReg; // Count eta regions increasing away from theta = 90 degrees.
    stateDigi.etaSectZsign = true; // Eta sector in -ve z half of Tracker.
  } else {
    stateDigi.etaSectID = iEtaReg - numEtaRegions_/2;
    stateDigi.etaSectZsign = false;
  }

  stateDigi.valid = true;

  return stateDigi;
}

//=== Implement NPAR-specific code called by getDigiStateIn(...).

template <>
void KFParamsCombCallHLS::getDigiStateInUtil<4>(const vector<double>& helixParams, const TMatrixD& cov, KalmanHLS::KFstate<4>& stateDigi) const {}

template <>
void KFParamsCombCallHLS::getDigiStateInUtil<5>(const vector<double>& helixParams, const TMatrixD& cov, KalmanHLS::KFstate<5>& stateDigi) const {
                                                                 
  double d0  = helixParams[4];
  double cov_d0_d0    = cov[4][4];
  double cov_inv2R_d0 = cov[0][4];                   
  double cov_phi0_d0  = cov[1][4];

  stateDigi.d0  = d0*d0_Mult_;
  stateDigi.cov_44 = cov_d0_d0 * d0_Mult_ * d0_Mult_;
  stateDigi.cov_04 = cov_inv2R_d0 * inv2R_Mult_ * d0_Mult_;
  stateDigi.cov_14 = cov_phi0_d0 * phiMult_ * d0_Mult_;
  KalmanHLS::CHECK_AP::checkCalc("helix4", stateDigi.d0   , d0*d0_Mult_        , 9.9e9, 0.1);
  KalmanHLS::CHECK_AP::checkCalc("C44_old", stateDigi.cov_44, cov_d0_d0       * d0_Mult_      * d0_Mult_);
  KalmanHLS::CHECK_AP::checkCalc("C04_old", stateDigi.cov_04, cov_inv2R_d0    * inv2R_Mult_   * d0_Mult_);
  KalmanHLS::CHECK_AP::checkCalc("C14_old", stateDigi.cov_14, cov_phi0_d0     * phiMult_      * d0_Mult_);
}

//=== Convert digitized ourput KF state to floating point for both NPAR = 4 & 5 param helix fits.

template<unsigned int NPAR>
const KalmanState* KFParamsCombCallHLS::getStateOut(const KalmanState* stateIn, const StubCluster* stubCluster, const KalmanHLS::KFstate<NPAR>& stateOutDigi, const KalmanHLS::KFselect<NPAR>& selectOutDigi) {
  // Convert digitized helix state to floating point one.
  // Also copy some info directly from input floating point to output floating point state, if unchanged.

  // Fill arguments of L1KalmanComb::mkState(), which is called to make a KalmanState object.
  const L1track3D& candidate = stateIn->candidate();
  unsigned int n_skipped     = stateOutDigi.nSkippedLayers;
  unsigned int kLayer_next   = stateOutDigi.layerID; // Unchanged by KF updator.
  unsigned int layerId       = stateIn->layerId();
  const KalmanState* last_state = stateIn; 

  // Factors to convert digitized helix params to floating ones are inverse of those in getDigiStateIn().
  vector<double> x(NPAR); // helix params
  x[0] = (double(stateOutDigi.inv2R) + 0.5 / pow(2, stateOutDigi.inv2R.width - stateOutDigi.inv2R.iwidth)) / inv2R_Mult_;
  x[1] = (double(stateOutDigi.phi0)  + 0.5 / pow(2, stateOutDigi.phi0.width  - stateOutDigi.phi0.iwidth )) / phiMult_;
  x[2] = (double(stateOutDigi.tanL)  + 0.5 / pow(2, stateOutDigi.tanL.width  - stateOutDigi.tanL.iwidth ));
  x[3] = (double(stateOutDigi.z0)    + 0.5 / pow(2, stateOutDigi.z0.width    - stateOutDigi.z0.iwidth   )) / rMult_;

  TMatrixD pxx(NPAR,NPAR); // helix covariance matrix
  pxx[0][0] = (double(stateOutDigi.cov_00) + 0.5 / pow(2, stateOutDigi.cov_00.width - stateOutDigi.cov_00.iwidth)) / (inv2R_Mult_ * inv2R_Mult_);
  pxx[1][1] = (double(stateOutDigi.cov_11) + 0.5 / pow(2, stateOutDigi.cov_11.width - stateOutDigi.cov_11.iwidth)) / (phiMult_ * phiMult_);
  pxx[2][2] = (double(stateOutDigi.cov_22) + 0.5 / pow(2, stateOutDigi.cov_22.width - stateOutDigi.cov_22.iwidth));
  pxx[3][3] = (double(stateOutDigi.cov_33) + 0.5 / pow(2, stateOutDigi.cov_33.width - stateOutDigi.cov_33.iwidth)) / (rMult_ * rMult_);
  pxx[0][1] = (double(stateOutDigi.cov_01) + 0.5 / pow(2, stateOutDigi.cov_01.width - stateOutDigi.cov_01.iwidth)) / (inv2R_Mult_ * phiMult_);
  pxx[1][0] = pxx[0][1];
  pxx[2][3] = (double(stateOutDigi.cov_23) + 0.5 / pow(2, stateOutDigi.cov_23.width - stateOutDigi.cov_23.iwidth)) / (rMult_);
  pxx[3][2] = pxx[2][3];

  this->getStateOutUtil(stateOutDigi, selectOutDigi, x, pxx);

  TMatrixD K(nPar_,2); // KF gain matrix - don't provide, as can't be used?
  TMatrixD dcov(2,2);  // Stub (phi,z) position covariance matrix - don't provide as can't be used?
  const StubCluster* stubcl = stubCluster;
  double chi2rphi = (double(stateOutDigi.chiSquaredRphi) + 0.5 / pow(2, stateOutDigi.chiSquaredRphi.width - stateOutDigi.chiSquaredRphi.iwidth));
  double chi2rz   = (double(stateOutDigi.chiSquaredRz)   + 0.5 / pow(2, stateOutDigi.chiSquaredRz.width - stateOutDigi.chiSquaredRz.iwidth));

  const KalmanState* ks = this->mkState(candidate, n_skipped, kLayer_next, layerId, last_state,
				        x, pxx, K, dcov, stubcl, chi2rphi, chi2rz);

#ifdef PRINT_HLSARGS
  std::cout<<std::hex<<"OUTPUT hitPattern="<<ks->hitPattern()<<std::dec<<std::endl;
#endif

  // Commented out, as thse flags no longer generated by KF HLS.
  //  (const_cast<KalmanState*>(ks))->setHLSselect(int(selectOutDigi.mBin_fit) + getSettings()->houghNbinsPt()/2, int(selectOutDigi.cBin_fit) + getSettings()->houghNbinsPhi()/2, bool(selectOutDigi.consistent));

#ifdef IRT_DEBUG
  if (ks->candidate().getMatchedTP() != nullptr) {
    unsigned int iPhiSec = ks->candidate().iPhiSec();
    float phiCentreSec0 = -M_PI/float(getSettings()->numPhiNonants()) + M_PI/float(getSettings()->numPhiSectors());
    float phiSec = 2.*M_PI * float(iPhiSec) / float(getSettings()->numPhiSectors()) + phiCentreSec0; // Centre of sector in phi
    double inv2R = x[0]; // Half inverse radius of curvature.
    double phi0  = x[1]; // Measured with respect to centre of phi sector.
    double tanL  = x[2];
    double z0    = x[3];
    cout<<"KF Output track (digi): q/pt = "<<inv2R/inv2R_Mult_/(0.5*getSettings()->invPtToInvR())<<" phi0 = "<<phi0/phiMult_<<endl;
    cout<<"          truth: q/pt = "<<ks->candidate().getMatchedTP()->qOverPt()<<" phi0 = "<<reco::deltaPhi(ks->candidate().getMatchedTP()->phi0(),phiSec)<<endl;
  }
#endif

  return ks;
}

//=== Implement NPAR-specific code call by getStateOut(...).

template <>
void KFParamsCombCallHLS::getStateOutUtil<4>(const KalmanHLS::KFstate<4>& stateOutDigi, const KalmanHLS::KFselect<4>& selectOutDigi, vector<double>& x, TMatrixD& pxx) {
  // Store the extra info provided by the HLS updator about whether the state passes cuts.
  selectOutDigi4_ = selectOutDigi;
}

template <>
void KFParamsCombCallHLS::getStateOutUtil<5>(const KalmanHLS::KFstate<5>& stateOutDigi, const KalmanHLS::KFselect<5>& selectOutDigi, vector<double>& x, TMatrixD& pxx) {
  x[4] = (double(stateOutDigi.d0) + 0.5 / pow(2, stateOutDigi.d0.width - stateOutDigi.d0.iwidth)) / d0_Mult_;

  pxx[4][4] = (double(stateOutDigi.cov_44) + 0.5 / pow(2, stateOutDigi.cov_44.width - stateOutDigi.cov_44.iwidth)) / (d0_Mult_ * d0_Mult_);
  pxx[0][4] = (double(stateOutDigi.cov_04) + 0.5 / pow(2, stateOutDigi.cov_04.width - stateOutDigi.cov_04.iwidth)) / (inv2R_Mult_ * d0_Mult_);
  pxx[1][4] = (double(stateOutDigi.cov_14) + 0.5 / pow(2, stateOutDigi.cov_14.width - stateOutDigi.cov_14.iwidth)) / (phiMult_ * d0_Mult_);

  // Store the extra info provided by the HLS updator about whether the state passes cuts.
  selectOutDigi5_ = selectOutDigi;
}

//=== This is identical to version in KFParamsComb, deciding if a state passes cuts,
//=== except that it also checks the cut decisions produced by the HLS KalmanUpdate.

bool KFParamsCombCallHLS::isGoodState( const KalmanState &state ) const
{

#ifdef IRT_DEBUG
  bool trueTP = (state.candidate().getMatchedTP() != nullptr);
  bool vTrueTP = (trueTP && state.candidate().getPurity() < 0.99 && state.candidate().pt() > 3);
  cout<<"GOOD CHECK "<<trueTP<<" "<<vTrueTP<<endl;
#endif

  // Check if this state passes cuts using standard C++ KF code.
  bool goodState = KFParamsComb::isGoodState(state);

  // Check if this state passes cuts using HLS KF xode.
  bool goodState_HLS;
  if (nPar_ == 4) {
    goodState_HLS = selectOutDigi4_.z0Cut && selectOutDigi4_.ptCut && selectOutDigi4_.chiSquaredCut && selectOutDigi4_.sufficientPScut;
  } else {
    goodState_HLS = selectOutDigi5_.z0Cut && selectOutDigi5_.ptCut && selectOutDigi5_.chiSquaredCut && selectOutDigi5_.sufficientPScut && selectOutDigi5_.d0Cut;
  }

  unsigned nStubLayers = state.nStubLayers();
  double pt=fabs( getSettings()->invPtToInvR() / (2*state.xa()[INV2R]) ); 
  double z0=fabs( state.xa()[Z0] ); 

  // Check if this state passes Algo50 duplicate removal cuts.
  // (Approximate since it negelects "rescue" step of Algo50).
  //bool algo50_HLS = true;
  if (nStubLayers > 3) {
    // algo50_HLS = (selectOutDigi_.consistent && selectOutDigi_.sectorCut);
    // Debug printout to debug Algo50. To be used with "Cheat" cfg option. 
    // And print fittedTrack.consistent() from TMTrackProducer.cc.
    //
    // cout<<"algo50: HT consistent="<<selectOutDigi_.consistent<<" sector consistent="<<selectOutDigi_.sectorCut
    //   <<" mbin = "<<(state.candidate().getCellLocationHT().first-floor(getSettings()->houghNbinsPt()/2))
    //   <<" vs "<<selectOutDigi_.mBin_fit
    //   <<" cbin = "<<state.candidate().getCellLocationHT().second-floor(getSettings()->houghNbinsPhi()/2)
    //   <<" vs "<<selectOutDigi_.cBin_fit<<endl;
  }

  // Check if HLS & C++ KF agree ...

  if (settings_->kalmanDebugLevel() >= 1) {
    if (goodState && not goodState_HLS) {
      // Errors caused by small precision errors in chi2 cut value.
      if (nPar_ == 4) {
	cout<<"ERROR: KF HLS incorrectly rejected state "<<nStubLayers<<" "<<selectOutDigi4_.z0Cut<<" "<<selectOutDigi4_.ptCut<<" "<<selectOutDigi4_.chiSquaredCut<<" "<<selectOutDigi4_.sufficientPScut<<" : chi2="<<state.chi2()<<" pt="<<pt<<" 1/2R="<<state.xa()[INV2R]<<" z0="<<state.xa()[Z0]<<endl;
      } else {
	cout<<"ERROR: KF HLS incorrectly rejected state "<<nStubLayers<<" "<<selectOutDigi5_.z0Cut<<" "<<selectOutDigi5_.ptCut<<" "<<selectOutDigi5_.chiSquaredCut<<" "<<selectOutDigi5_.sufficientPScut<<" "<<selectOutDigi5_.d0Cut<<" : chi2="<<state.chi2()<<" pt="<<pt<<" 1/2R="<<state.xa()[INV2R]<<" z0="<<state.xa()[Z0]<<" d0="<<state.xa()[D0]<<endl;
      }
    } else if (not goodState && goodState_HLS) {
      // Failures here usually caused by failing chi2 cut by miniscule amount.
      if (nPar_ == 4) {
	cout<<"ERROR: KF HLS incorrectly kept state "<<nStubLayers<<" "<<selectOutDigi4_.z0Cut<<" "<<selectOutDigi4_.ptCut<<" "<<selectOutDigi4_.chiSquaredCut<<" "<<selectOutDigi4_.sufficientPScut<<" : chi2="<<state.chi2()<<" pt="<<pt<<" 1/2R="<<state.xa()[INV2R]<<" z0="<<state.xa()[Z0]<<endl;
      } else {
	cout<<"ERROR: KF HLS incorrectly kept state "<<nStubLayers<<" "<<selectOutDigi5_.z0Cut<<" "<<selectOutDigi5_.ptCut<<" "<<selectOutDigi5_.chiSquaredCut<<" "<<selectOutDigi5_.sufficientPScut<<" "<<selectOutDigi5_.d0Cut<<" : chi2="<<state.chi2()<<" pt="<<pt<<" 1/2R="<<state.xa()[INV2R]<<" z0="<<state.xa()[Z0]<<" d0="<<state.xa()[D0]<<endl;
      }
#ifdef IRT_DEBUG
    } else {
      cout<<"KF HLS CUTS GOOD "<<nStubLayers<<" "<<goodState_HLS<<" algo50 "<<algo50_HLS<<endl;
#endif
    }

#ifdef IRT_DEBUG
    if (trueTP) cout<<"L1K GOOD STATE CHECK: vtrue="<<vTrueTP<<" good="<<goodState_HLS<<" nLay="<<nStubLayers<<" chi2="<<state.chi2()<<" pt="<<pt<<" z0="<<state.xa()[Z0]<<" skip="<<state.nSkippedLayers()<<endl;
#endif
  }

  //return goodState;

  return goodState_HLS;
  //return goodState_HLS && algo50_HLS; // includes approximate algo50 duplicate removal, without rescue step.
}

}
