#ifndef __L1fittedTrack_H__
#define __L1fittedTrack_H__

#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1trackBase.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1track3D.h"
#include "L1Trigger/TrackFindingTMTT/interface/Settings.h"
#include "L1Trigger/TrackFindingTMTT/interface/Utility.h"
#include "L1Trigger/TrackFindingTMTT/interface/TP.h"
#include "L1Trigger/TrackFindingTMTT/interface/Stub.h"
#include "L1Trigger/TrackFindingTMTT/interface/Sector.h"
#include "L1Trigger/TrackFindingTMTT/interface/HTrphi.h"
#include "L1Trigger/TrackFindingTMTT/interface/DigitalTrack.h"
#include "L1Trigger/TrackFindingTMTT/interface/KFTrackletTrack.h"

#include <vector>
#include <set>
#include <utility>
#include <string>

using namespace std;

//=== This represents a fitted L1 track candidate found in 3 dimensions.
//=== It gives access to the fitted helix parameters & chi2 etc.
//=== It also calculates & gives access to associated truth particle (Tracking Particle) if any.
//=== It also gives access to the 3D hough-transform track candidate (L1track3D) on which the fit was run.

namespace TMTT {

class L1fittedTrack : public L1trackBase {

public:

  // Store a new fitted track, specifying the input Hough transform track, the stubs used for the fit,
  // bit-encoded hit layer pattern (numbered by increasing distance from origin),
  // the fitted helix parameters & chi2,
  // and the number of helix parameters being fitted (=5 if d0 is fitted, or =4 if d0 is not fitted).
  // And if track fit declared this to be a valid track (enough stubs left on track after fit etc.).
  L1fittedTrack(const Settings* settings, const L1track3D& l1track3D, const vector<const Stub*>& stubs, 
		unsigned int hitPattern,
                float qOverPt, float d0, float phi0, float z0, float tanLambda, 
                float chi2rphi, float chi2rz, unsigned int nHelixParam, bool accepted = true) :
    L1trackBase(),
    settings_(settings),
    l1track3D_(l1track3D), stubs_(stubs), hitPattern_(hitPattern),
    qOverPt_(qOverPt), d0_(d0), phi0_(phi0), z0_(z0), tanLambda_(tanLambda),
    chi2rphi_(chi2rphi), chi2rz_(chi2rz), 
    done_bcon_(false), qOverPt_bcon_(qOverPt), d0_bcon_(d0), phi0_bcon_(phi0), chi2rphi_bcon_(chi2rphi),
    nHelixParam_(nHelixParam),
    iPhiSec_(l1track3D.iPhiSec()), iEtaReg_(l1track3D.iEtaReg()), 
    optoLinkID_(l1track3D.optoLinkID()), accepted_(accepted),
    nSkippedLayers_(0), numUpdateCalls_(0), numIterations_(0), 
    digitizedTrack_(false), digitalTrack_(settings)
  {
    // Doesn't make sense to assign stubs to track if fitter rejected it.
    if (! accepted) stubs_.clear();
    nLayers_   = Utility::countLayers(settings, stubs); // Count tracker layers these stubs are in
    matchedTP_ = Utility::matchingTP(settings, stubs, nMatchedLayers_, matchedStubs_); // Find associated truth particle & calculate info about match.
    // Set d0 = 0 for 4 param fit, in case fitter didn't do it.
    if (nHelixParam == 4) {
      d0_ = 0.;
      d0_bcon_ = 0.;
    }
    if (! settings->hybrid()) {
      secTmp_.init(settings, iPhiSec_, iEtaReg_); //Sector class used to check if fitted track trajectory is in expected sector.
      htRphiTmp_.init(settings, iPhiSec_, iEtaReg_, secTmp_.etaMin(), secTmp_.etaMax(), secTmp_.phiCentre()); // HT class used to identify HT cell that corresponds to fitted helix parameters.
    }
    this->setConsistentHTcell(); 
  }

  L1fittedTrack() : L1trackBase() {}; // Creates track object, but doesn't set any variables.

  ~L1fittedTrack() {}

  //--- Optionally set track helix params & chi2 if beam-spot constraint is used (for 5-parameter fit).
  void setBeamConstr(float qOverPt_bcon, float phi0_bcon, float chi2rphi_bcon) {
    done_bcon_ = true;  qOverPt_bcon_ = qOverPt_bcon;  d0_bcon_ = 0.0, phi0_bcon_ = phi0_bcon; chi2rphi_bcon_ = chi2rphi_bcon;
  }

  //--- Set/get additional info about fitted track that is specific to individual track fit algorithms (KF, LR, chi2)
  //--- and is used for debugging/histogramming purposes.
  
  void setInfoKF( unsigned int nSkippedLayers, unsigned int numUpdateCalls ) {
    nSkippedLayers_ = nSkippedLayers;
    numUpdateCalls_ = numUpdateCalls;
  }
  void setInfoKF( unsigned int nSkippedLayers, unsigned int numUpdateCalls, bool consistentHLS ) {
    this->setInfoKF(nSkippedLayers_, numUpdateCalls_);
    // consistentCell_ = consistentHLS; // KF HLS code no longer calculates HT cell consistency.
  }
  void setInfoLR( int numIterations, std::string lostMatchingState, std::unordered_map< std::string, int > stateCalls ) {
    numIterations_ = numIterations; lostMatchingState_ = lostMatchingState; stateCalls_ = stateCalls;
  }
  void setInfoCHI2() {}

  void getInfoKF( unsigned int& nSkippedLayers, unsigned int& numUpdateCalls ) const {
    nSkippedLayers = nSkippedLayers_;
    numUpdateCalls = numUpdateCalls_;
  }
  void getInfoLR( int& numIterations, std::string& lostMatchingState, std::unordered_map< std::string, int >& stateCalls ) const {
    numIterations = numIterations_; lostMatchingState = lostMatchingState_; stateCalls = stateCalls_;
  }
  void getInfoCHI2()                                const     {}

  //--- Convert fitted track to KFTrackletTrack format, for use with HYBRID.

  KFTrackletTrack returnKFTrackletTrack(){
    KFTrackletTrack trk_(getL1track3D(), getStubs(), getHitPattern(), qOverPt(), d0(), phi0(), z0(), tanLambda(), 
			 chi2rphi(),  chi2rz(), nHelixParam(), iPhiSec(), iEtaReg(), accepted());
    return trk_;
  }


  //--- Get the 3D Hough transform track candididate corresponding to the fitted track,
  //--- Provide direct access to some of the info it contains.

  // Get track candidate from HT (before fit).
  const L1track3D&            getL1track3D()          const  {return l1track3D_;}

  // Get stubs on fitted track (can differ from those on HT track if track fit kicked out stubs with bad residuals)
  const vector<const Stub*>&  getStubs()              const  {return stubs_;}  
  // Get number of stubs on fitted track.
  unsigned int                getNumStubs()           const  {return stubs_.size();}
  // Get number of tracker layers these stubs are in.
  unsigned int                getNumLayers()          const  {return nLayers_;}
  // Get number of stubs deleted from track candidate by fitter (because they had large residuals)
  unsigned int                getNumKilledStubs()        const  {return l1track3D_.getNumStubs() - this->getNumStubs();}
  // Get bit-encoded hit pattern (where layer number assigned by increasing distance from origin, according to layers track expected to cross).
  unsigned int                getHitPattern()        const  {return hitPattern_;}

  // Get Hough transform cell locations in units of bin number, corresponding to the fitted helix parameters of the track.
  // Always uses the beam-spot constrained helix params if they are available.
  // (If fitted track is outside HT array, it it put in the closest bin inside it).
  pair<unsigned int, unsigned int>  getCellLocationFit() const {return htRphiTmp_.getCell(this);}
  // Also get HT cell determined by Hough transform.
  pair<unsigned int, unsigned int>  getCellLocationHT()  const {return l1track3D_.getCellLocationHT();}

  //--- Get information about its association (if any) to a truth Tracking Particle.
  //--- Can differ from that of corresponding HT track, if track fit kicked out stubs with bad residuals.

  // Get best matching tracking particle (=nullptr if none).
  const TP*                   getMatchedTP()          const  {return matchedTP_;}
  // Get the matched stubs with this Tracking Particle
  const vector<const Stub*>&  getMatchedStubs()       const  {return matchedStubs_;}
  // Get number of matched stubs with this Tracking Particle
  unsigned int                getNumMatchedStubs()    const  {return matchedStubs_.size();}
  // Get number of tracker layers with matched stubs with this Tracking Particle
  unsigned int                getNumMatchedLayers()   const  {return nMatchedLayers_;}
  // Get purity of stubs on track (i.e. fraction matching best Tracking Particle)
  float                       getPurity()             const   {return getNumMatchedStubs()/float(getNumStubs());}
  // Get number of stubs matched to correct TP that were deleted from track candidate by fitter.
  unsigned int                getNumKilledMatchedStubs()  const  {
    unsigned int nStubCount = l1track3D_.getNumMatchedStubs();
    if (nStubCount > 0) { // Original HT track candidate did match a truth particle
      const TP* tp = l1track3D_.getMatchedTP();
      for (const Stub* s : stubs_) {
        set<const TP*> assTPs = s->assocTPs();
        if (assTPs.find(tp) != assTPs.end()) nStubCount--; // We found a stub matched to original truth particle that survived fit.
      }
    }
    return nStubCount;
  }

  //--- Get the fitted track helix parameters.

  float   qOverPt()      const  {return qOverPt_;}
  float   charge()       const  {return (qOverPt_ > 0  ?  1  :  -1);}
  float   invPt()        const  {return fabs(qOverPt_);}
  float   pt()           const  {return 1./(1.0e-6 + this->invPt());} // includes protection against 1/pt = 0.
  float   d0()           const  {return d0_;}
  float   phi0()         const  {return phi0_;}
  float   z0()           const  {return z0_;}
  float   tanLambda()    const  {return tanLambda_;}
  float   theta()        const  {return atan2(1., tanLambda_);} // Use atan2 to ensure 0 < theta < pi.
  float   eta()          const  {return -log(tan(0.5*this->theta()));}

  //--- Get the fitted helix parameters with beam-spot constraint.
  //--- If constraint not applied (e.g. 4 param fit) then these are identical to unconstrained values.

  bool   done_bcon()     const  {return done_bcon_;} // Was beam-spot constraint aplied?
  float  qOverPt_bcon()  const  {return qOverPt_bcon_;}
  float  charge_bcon()   const  {return (qOverPt_bcon_ > 0  ?  1  :  -1);}
  float  invPt_bcon()    const  {return fabs(qOverPt_bcon_);}
  float  pt_bcon()       const  {return 1./(1.0e-6 + this->invPt_bcon());}
  float  phi0_bcon()     const  {return phi0_bcon_;}

  // Phi and z coordinates at which track crosses "chosenR" values used by r-phi HT and rapidity sectors respectively. 
  // (Optionally with beam-spot constraint applied).
  float   phiAtChosenR(bool beamConstraint) const {
    if (beamConstraint) {
      return reco::deltaPhi(phi0_bcon_ - ((settings_->invPtToDphi() * settings_->chosenRofPhi()) * qOverPt_bcon_) - d0_bcon_/(settings_->chosenRofPhi()),  0.);
    } else {
      return reco::deltaPhi(phi0_ - ((settings_->invPtToDphi() * settings_->chosenRofPhi()) * qOverPt_) - d0_/(settings_->chosenRofPhi()),  0.);
    }
  }
  float   zAtChosenR()   const  {return (z0_ + (settings_->chosenRofZ()) * tanLambda_);} // neglects transverse impact parameter & track curvature.

  // Get the number of helix parameters being fitted (=5 if d0 is fitted or =4 if d0 is not fitted).
  float   nHelixParam()  const  {return nHelixParam_;}

  // Get the fit degrees of freedom, chi2 & chi2/DOF (also in r-phi & r-z planes).
  unsigned int numDOF()      const  {return 2*this->getNumStubs() - nHelixParam_;}
  unsigned int numDOFrphi()  const  {return this->getNumStubs() - (nHelixParam_ - 2);}
  unsigned int numDOFrz(  )  const  {return this->getNumStubs() - 2;}
  float   chi2rphi()     const  {return chi2rphi_;}
  float   chi2rz()       const  {return chi2rz_;}
  float   chi2()         const  {return chi2rphi_ + chi2rz_;}
  float   chi2dof()      const  {return (this->chi2())/this->numDOF();}

  //--- Ditto, but if beam-spot constraint is applied.
  //--- If constraint not applied (e.g. 4 param fit) then these are identical to unconstrained values.
  unsigned int numDOF_bcon()      const  {return (this->numDOF() - 1);}
  unsigned int numDOFrphi_bcon()  const  {return (this->numDOFrphi() - 1);}
  float   chi2rphi_bcon()     const  {return chi2rphi_bcon_;}
  float   chi2_bcon()         const  {return chi2rphi_bcon_ + chi2rz_;}
  float   chi2dof_bcon()      const  {return (this->chi2_bcon())/this->numDOF_bcon();}

  //--- Get phi sector and eta region used by track finding code that this track is in.
  unsigned int iPhiSec() const  {return iPhiSec_;}
  unsigned int iEtaReg() const  {return iEtaReg_;}

  //--- Opto-link ID used to send this track from HT to Track Fitter
  unsigned int optoLinkID() const {return optoLinkID_;}

  //--- Get whether the track has been rejected or accepted by the fit

  bool accepted()  const  {
    return accepted_;
  }

  // Comparitor useful for sorting tracks by q/Pt using std::sort().
  static bool qOverPtSortPredicate(const L1fittedTrack& t1, const L1fittedTrack t2) { return t1.qOverPt() < t2.qOverPt(); }

  //--- Functions to help eliminate duplicate tracks.

  // Is the fitted track trajectory should lie within the same HT cell in which the track was originally found?
  bool consistentHTcell() const {return consistentCell_;}

  // Determine if the fitted track trajectory should lie within the same HT cell in which the track was originally found?
  void setConsistentHTcell() {
    //return (max(fabs(this->deltaM()), fabs(this->deltaC())) < 0.5);
    // Use helix params with beam-spot constaint if done in case of 5 param fit.

    pair<unsigned int, unsigned int> htCell = this->getCellLocationHT();
    bool consistent = (htCell == this->getCellLocationFit()); 

    if (l1track3D_.mergedHTcell()) {
      // If this is a merged cell, check other elements of merged cell.
      pair<unsigned int, unsigned int> htCell10( htCell.first + 1, htCell.second);
      pair<unsigned int, unsigned int> htCell01( htCell.first    , htCell.second + 1);
      pair<unsigned int, unsigned int> htCell11( htCell.first + 1, htCell.second + 1);
      if (htCell10 == this->getCellLocationFit()) consistent = true; 
      if (htCell01 == this->getCellLocationFit()) consistent = true; 
      if (htCell11 == this->getCellLocationFit()) consistent = true; 
    }

    consistentCell_ = consistent;
  }

  // Is the fitted track trajectory within the same (eta,phi) sector of the HT used to find it?
  bool consistentSector() const {
    bool insidePhi = (fabs(reco::deltaPhi(this->phiAtChosenR(done_bcon_), secTmp_.phiCentre())) < secTmp_.sectorHalfWidth());
    bool insideEta = (this->zAtChosenR() > secTmp_.zAtChosenR_Min() && this->zAtChosenR() < secTmp_.zAtChosenR_Max());
    return (insidePhi && insideEta);
  }

  // Digitize track and degrade helix parameter resolution according to effect of digitisation.
  void digitizeTrack(const string& fitterName);

  // Access to detailed info about digitized track
  const DigitalTrack&             digitaltrack() const { return      digitalTrack_;}

private:

  //--- Configuration parameters
  const Settings*                    settings_;

  //--- The 3D hough-transform track candidate which was fitted.
  L1track3D             l1track3D_;

  //--- The stubs on the fitted track (can differ from those on HT track if fit kicked off stubs with bad residuals)
  vector<const Stub*>   stubs_;
  unsigned int          nLayers_;

  //--- Bit-encoded hit pattern (where layer number assigned by increasing distance from origin, according to layers track expected to cross).
  unsigned int          hitPattern_;

  //--- The fitted helix parameters and fit chi-squared.
  float qOverPt_;
  float d0_;
  float phi0_;
  float z0_;
  float tanLambda_;
  float chi2rphi_;
  float chi2rz_;

  //--- Ditto with beam-spot constraint applied in case of 5-parameter fit, plus boolean to indicate
  bool  done_bcon_;
  float qOverPt_bcon_;
  float d0_bcon_;
  float phi0_bcon_;
  float chi2rphi_bcon_;

  //--- The number of helix parameters being fitted (=5 if d0 is fitted or =4 if d0 is not fitted).
  unsigned int nHelixParam_;

  //--- Phi sector and eta region used track finding code that this track was in.
  unsigned int iPhiSec_;
  unsigned int iEtaReg_;
  //--- Opto-link ID from HT to Track Fitter.
  unsigned int optoLinkID_;

  //--- Information about its association (if any) to a truth Tracking Particle.
  const TP*             matchedTP_;
  vector<const Stub*>   matchedStubs_;
  unsigned int          nMatchedLayers_;

  //--- Has the track fit declared this to be a valid track?
  bool accepted_;

  //--- Sector class used to check if fitted track trajectory is in same sector as HT used to find it.
  Sector secTmp_;
  //--- r-phi HT class used to determine HT cell location that corresponds to fitted track helix parameters.
  HTrphi htRphiTmp_;

  //--- Info specific to KF fitter.
  unsigned int nSkippedLayers_;
  unsigned int numUpdateCalls_;
  //--- Info specific to LR fitter.
  int numIterations_;
  std::string lostMatchingState_;
  std::unordered_map< std::string, int > stateCalls_;

  bool digitizedTrack_;
  DigitalTrack                          digitalTrack_; // Class used to digitize track if required.

  bool consistentCell_;
};

}

#endif
