#ifndef __L1track3D_H__
#define __L1track3D_H__

#include "DataFormats/Math/interface/deltaPhi.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1trackBase.h"
#include "L1Trigger/TrackFindingTMTT/interface/Settings.h"
#include "L1Trigger/TrackFindingTMTT/interface/Utility.h"
#include "L1Trigger/TrackFindingTMTT/interface/TP.h"
#include "L1Trigger/TrackFindingTMTT/interface/Stub.h"
#include "L1Trigger/TrackFindingTMTT/interface/Sector.h"
#include "L1Trigger/TrackFindingTMTT/interface/HTrphi.h"

#include <vector>
#include <string>
#include <unordered_set>
#include <utility>

using namespace std;

//=== L1 track candidate found in 3 dimensions.
//=== Gives access to all stubs on track and to its 3D helix parameters.
//=== Also calculates & gives access to associated truth particle (Tracking Particle) if any.

namespace TMTT {

class L1track3D : public L1trackBase {

public:

  L1track3D(const Settings* settings, const vector<const Stub*>& stubs,
            pair<unsigned int, unsigned int> cellLocationHT, pair<float, float> helixRphi, pair<float, float> helixRz, float helixD0,
      unsigned int iPhiSec, unsigned int iEtaReg, unsigned int optoLinkID, bool mergedHTcell) : 
    L1trackBase(),
    settings_(settings),
    stubs_(stubs), 
    cellLocationHT_(cellLocationHT), helixRphi_(helixRphi), helixRz_  (helixRz),  helixD0_(helixD0),
    iPhiSec_(iPhiSec), iEtaReg_(iEtaReg), optoLinkID_(optoLinkID), mergedHTcell_(mergedHTcell),
    seedLayerType_(999), seedPS_(999)
  {
    nLayers_   = Utility::countLayers(settings, stubs); // Count tracker layers these stubs are in
    matchedTP_ = Utility::matchingTP(settings, stubs, nMatchedLayers_, matchedStubs_); // Find associated truth particle & calculate info about match.
  }

  L1track3D(const Settings* settings, const vector<const Stub*>& stubs,
            pair<unsigned int, unsigned int> cellLocationHT, pair<float, float> helixRphi, pair<float, float> helixRz,
      unsigned int iPhiSec, unsigned int iEtaReg, unsigned int optoLinkID, bool mergedHTcell) : 
    L1track3D(settings, stubs, cellLocationHT,  helixRphi,  helixRz,  0.0, iPhiSec,  iEtaReg,  optoLinkID,  mergedHTcell){}

  L1track3D() : L1trackBase() {}; // Creates track object, but doesn't set any variables.

  ~L1track3D() {}

  //--- Set/get optional info for tracklet tracks.

  // Tracklet seeding layer pair (from FPGATracklet::seedIndex())
  // 0-7 = "L1L2","L2L3","L3L4","L5L6","D1D2","D3D4","L1D1","L2D1"
  void setSeedLayerType(unsigned int seedLayerType) {seedLayerType_ = seedLayerType;}
  unsigned int seedLayerType() const {return seedLayerType_;}

  // Tracklet seed stub pair uses PS modules (from FPGATracket::PSseed())
  void setSeedPS(unsigned int seedPS) {seedPS_ = seedPS;}
  unsigned int seedPS() const {return seedPS_;}

  // Best stub (stub with smallest Phi residual in each layer/disk)
  void setBestStubs(std::unordered_set<const Stub*> bestStubs) {bestStubs_ = bestStubs;}
  std::unordered_set<const Stub*> bestStubs() const {return bestStubs_;}

  //--- Get information about the reconstructed track.

  // Get stubs on track candidate.
  const vector<const Stub*>&        getStubs()              const  {return stubs_;}  
  // Get number of stubs on track candidate.
  unsigned int                      getNumStubs()           const  {return stubs_.size();}
  // Get number of tracker layers these stubs are in.
  unsigned int                      getNumLayers()          const  {return nLayers_;}
  // Get cell location of track candidate in r-phi Hough Transform array in units of bin number.
  pair<unsigned int, unsigned int>  getCellLocationHT()     const  {return cellLocationHT_;}
  // The two conventionally agreed track helix parameters relevant in r-phi plane. i.e. (q/Pt, phi0)
  pair<float, float>                getHelixRphi()          const  {return helixRphi_;}
  // The two conventionally agreed track helix parameters relevant in r-z plane. i.e. (z0, tan_lambda)
  pair<float, float>                getHelixRz()            const  {return helixRz_;}

  //--- Return chi variables, (both digitized & undigitized), which are the stub coords. relative to track.

  vector<float> getChiPhi() {
    vector<float> result;
    for (const Stub* s: stubs_) {
      float chi_phi = reco::deltaPhi(s->phi(), this->phi0() - s->r() * this->qOverPt() * settings_->invPtToDphi());
      result.push_back(chi_phi);
    }
    return result;
  }

  vector<int> getChiPhiDigi() {	  
    vector<int> result;
    static const float phiMult = pow(2, settings_->phiSBits()) / settings_->phiSRange();
    for (const float& chi_phi: this->getChiPhi()) {
      int iDigi_chi_phi = floor(chi_phi * phiMult);
      result.push_back(iDigi_chi_phi);
    }
    return result;
  }

  vector<float> getChiZ() {
    vector<float> result;
    for (const Stub* s: stubs_) {
      float chi_z = s->z() - (this->z0() + s->r()*this->tanLambda());
      result.push_back(chi_z);
    }
    return result;
  }

  vector<int> getChiZDigi() {	  
    vector<int> result;
    static const float zMult = pow(2, settings_->zBits()) / settings_->zRange();
    for (const float& chi_z: this->getChiZ()) {
      int iDigi_chi_z = floor(chi_z * zMult);
      result.push_back(iDigi_chi_z);
    }
    return result;
  }

  //--- User-friendlier access to the helix parameters. 

  float   qOverPt()    const  {return helixRphi_.first;}
  float   charge()     const  {return (this->qOverPt() > 0  ?  1  :  -1);} 
  float   invPt()      const  {return fabs(this->qOverPt());}
  float   pt()         const  {return 1./(1.0e-6 + this->invPt());} // includes protection against 1/pt = 0.
  float   d0()         const  {return helixD0_;} // Hough transform assumes d0 = 0.
  float   phi0()       const  {return helixRphi_.second;}
  float   z0()         const  {return helixRz_.first;}
  float   tanLambda()  const  {return helixRz_.second;}
  float   theta()      const  {return atan2(1., this->tanLambda());} // Use atan2 to ensure 0 < theta < pi.
  float   eta()        const  {return -log(tan(0.5*this->theta()));}

  // Phi and z coordinates at which track crosses "chosenR" values used by r-phi HT and rapidity sectors respectively.
  float   phiAtChosenR() const  {return reco::deltaPhi(this->phi0() - (settings_->invPtToDphi() * settings_->chosenRofPhi()) * this->qOverPt(),  0.);}
  float   zAtChosenR()   const  {return (this->z0() + (settings_->chosenRofZ()) * this->tanLambda());} // neglects transverse impact parameter & track curvature.

  //--- Get phi sector and eta region used by track finding code that this track is in.
  unsigned int iPhiSec() const  {return iPhiSec_;}
  unsigned int iEtaReg() const  {return iEtaReg_;}

  //--- Opto-link ID used to send this track from HT to Track Fitter
  unsigned int optoLinkID() const {return optoLinkID_;}

  //--- Was this track produced from a marged HT cell (e.g. 2x2)?
  bool mergedHTcell() const {return mergedHTcell_;}

  //--- Get information about its association (if any) to a truth Tracking Particle.

  // Get best matching tracking particle (=nullptr if none).
  const TP*                  getMatchedTP()          const   {return matchedTP_;}
  // Get the matched stubs with this Tracking Particle
  const vector<const Stub*>& getMatchedStubs()       const   {return matchedStubs_;}
  // Get number of matched stubs with this Tracking Particle
  unsigned int               getNumMatchedStubs()    const   {return matchedStubs_.size();}
  // Get number of tracker layers with matched stubs with this Tracking Particle 
  unsigned int               getNumMatchedLayers()   const   {return nMatchedLayers_;}
  // Get purity of stubs on track candidate (i.e. fraction matching best Tracking Particle)
  float                      getPurity()             const   {return getNumMatchedStubs()/float(getNumStubs());}

  //--- For debugging purposes.

  // Remove incorrect stubs from the track using truth information.
  // Also veto tracks where the HT cell estimated from the true helix parameters is inconsistent with the cell the HT found the track in, (since probable duplicates).
  // Also veto tracks that match a truth particle not used for the algo efficiency measurement.
  // Return a boolean indicating if the track should be kept. (i.e. Is genuine & non-duplicate).
  bool cheat() {
    bool keep = false;

    vector<const Stub*> stubsSel;
    if (matchedTP_ != nullptr) { // Genuine track
      for (const Stub* s : stubs_) {
        const TP* tp = s->assocTP();
        if (tp != nullptr) {
          if (matchedTP_->index() == tp->index()) {
            stubsSel.push_back(s); // This stub was produced by same truth particle as rest of track, so keep it.
	  }
        }
      }
    }
    stubs_ = stubsSel;

    nLayers_   = Utility::countLayers(settings_, stubs_); // Count tracker layers these stubs are in
    matchedTP_ = Utility::matchingTP(settings_, stubs_, nMatchedLayers_, matchedStubs_); // Find associated truth particle & calculate info about match.

    bool genuine = (matchedTP_ != nullptr);

    if (genuine && matchedTP_->useForAlgEff()) {
      Sector secTmp;
      HTrphi htRphiTmp;
      secTmp.init(settings_, iPhiSec_, iEtaReg_); 
      htRphiTmp.init(settings_, iPhiSec_, iEtaReg_, secTmp.etaMin(), secTmp.etaMax(), secTmp.phiCentre()); 
      pair<unsigned int, unsigned int> trueCell = htRphiTmp.trueCell(matchedTP_);

      pair<unsigned int, unsigned int> htCell = this->getCellLocationHT();
      bool consistent = (htCell == trueCell); // If true, track is probably not a duplicate. 
      if (mergedHTcell_) {
	// If this is a merged cell, check other elements of merged cell.
	pair<unsigned int, unsigned int> htCell10( htCell.first + 1, htCell.second);
	pair<unsigned int, unsigned int> htCell01( htCell.first    , htCell.second + 1);
	pair<unsigned int, unsigned int> htCell11( htCell.first + 1, htCell.second + 1);
	if (htCell10 == trueCell) consistent = true; 
	if (htCell01 == trueCell) consistent = true; 
	if (htCell11 == trueCell) consistent = true; 
      }
      if (consistent) keep = true;
    }

    return keep; // Indicate if track should be kept.
  }


private:

  //--- Configuration parameters
  const Settings*                    settings_; 

  //--- Information about the reconstructed track.
  vector<const Stub*>                stubs_;
  unordered_set<const Stub*>         bestStubs_;
  unsigned int                       nLayers_;
  pair<unsigned int, unsigned int>   cellLocationHT_; 
  pair<float, float>                 helixRphi_; 
  pair<float, float>                 helixRz_; 
  float                              helixD0_;
  unsigned int                       iPhiSec_;
  unsigned int                       iEtaReg_; 
  unsigned int                       optoLinkID_;
  bool                               mergedHTcell_;

  //--- Optional info used for tracklet tracks.
  unsigned int                       seedLayerType_;
  unsigned int                       seedPS_;

  //--- Information about its association (if any) to a truth Tracking Particle.
  const TP*                          matchedTP_;
  vector<const Stub*>                matchedStubs_;
  unsigned int                       nMatchedLayers_;
};

}

#endif
