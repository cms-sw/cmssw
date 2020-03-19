/**
 * This is the interface between the C++ KF framework in CMSSW and the HLS code.
 * It is identical to KF4ParamsComb, except that the state updator is modified 
 *
 * All variable names & equations come from Fruhwirth KF paper
 * http://dx.doi.org/10.1016/0168-9002%2887%2990887-4
 * 
 * Author: Ian Tomalin
 */
 
#ifndef __KFPARAMSCOMBCALLHLS__
#define __KFPARAMSCOMBCALLHLS__
 
#include "L1Trigger/TrackFindingTMTT/interface/KFParamsComb.h"
#include "L1Trigger/TrackFindingTMTT/interface/HLS/KFpragmaOpts.h"
#include "L1Trigger/TrackFindingTMTT/interface/HLS/KFconstants.h"
#include "L1Trigger/TrackFindingTMTT/interface/HLS/KFstub.h"
#include "L1Trigger/TrackFindingTMTT/interface/HLS/KFstate.h"

using namespace std;

namespace TMTT {

class TP; 
class KalmanState;
class StubCluster;

class KFParamsCombCallHLS : public KFParamsComb {

public:

  KFParamsCombCallHLS(const Settings* settings, const uint nPar, const string &fitterName ); 

  virtual ~KFParamsCombCallHLS(){}

  // Update KF helix params with this stub.
  // (Override KF state updator in L1KalmanComb with version suitable for HLS).
  const KalmanState *kalmanUpdate( unsigned skipped, unsigned layer, const StubCluster* stubCluster, const KalmanState &stateIn, const TP *);

  // Print summary info to help tune bit ranges of HLS calculations.
  virtual void endJob() {KalmanHLS::CHECK_AP::printCheckMap();}

protected:

  // Is this HLS code?
  virtual bool isHLS() {return true;};

private:

  // Get digital stub info that the KF VHDL injects into the KF state updater (Maxeller/HLS)
  KalmanHLS::KFstubC    getDigiStub(const StubCluster* stubCluster, const KalmanState* state);

  // Get digitised KF state info that the KF VHDL injects into the KF state updater (Maxeller/HLS),
  // both for NPAR = 4 & 5 param helix states.
  template <unsigned int NPAR> 
  KalmanHLS::KFstate<NPAR> getDigiStateIn(unsigned int skipped, unsigned int layer, const KalmanState* state) const;

  // Implement NPAR-specific code called by getDigiStateIn(...).
  template <unsigned int NPAR>
  void getDigiStateInUtil(const vector<double>& helixParams, const TMatrixD& cov, KalmanHLS::KFstate<NPAR>& stateDigi) const;

  // Convert digitized ourput KF state to floating point,
  // both for NPAR = 4 & 5 param helix states.
  template <unsigned int NPAR>
  const KalmanState* getStateOut(const KalmanState* stateIn, const StubCluster* stubCluster, const KalmanHLS::KFstate<NPAR>& stateOutDigi, const KalmanHLS::KFselect<NPAR>& selectOutDigi);

  // Implement NPAR-specific code call by getStateOut(...).
  template <unsigned int NPAR>
  void getStateOutUtil(const KalmanHLS::KFstate<NPAR>& stateOutDigi, const KalmanHLS::KFselect<NPAR>& selectOutDigi, vector<double>& x, TMatrixD& pxx);

  // This is identical to version in KFParamsComb, deciding if a state passes cuts,
  // except that it also checks the cut decisions produced by the HLS KalmanUpdate.
  bool isGoodState( const KalmanState &state ) const;

private:
  // Digitisation multipliers
  double rMult_;
  double zMult_;
  double phiMult_;
  double inv2R_Mult_;
  double d0_Mult_;
  // Reference radius in r-phi plane.
  double chosenRofPhi_;
  // Number of eta sectors.
  unsigned int numEtaRegions_;

  // Store the extra info provided by the HLS updator about whether the state passes cuts.
  KalmanHLS::KFselect<4> selectOutDigi4_;
  KalmanHLS::KFselect<5> selectOutDigi5_;
};

// Fully specialized templates must be declared outside class, but inside .h file, to ensure they are found.

template <>
void KFParamsCombCallHLS::getDigiStateInUtil<4>(const vector<double>& helixParams, const TMatrixD& cov, KalmanHLS::KFstate<4>& stateDigi) const;

template <>
void KFParamsCombCallHLS::getDigiStateInUtil<5>(const vector<double>& helixParams, const TMatrixD& cov, KalmanHLS::KFstate<5>& stateDigi) const;

template <>
void KFParamsCombCallHLS::getStateOutUtil<4>(const KalmanHLS::KFstate<4>& stateOutDigi, const KalmanHLS::KFselect<4>& selectOutDigi, vector<double>& x, TMatrixD& pxx);

template <>
void KFParamsCombCallHLS::getStateOutUtil<5>(const KalmanHLS::KFstate<5>& stateOutDigi, const KalmanHLS::KFselect<5>& selectOutDigi, vector<double>& x, TMatrixD& pxx);
}

#endif
