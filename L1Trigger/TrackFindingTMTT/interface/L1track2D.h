#ifndef __L1track2D_H__
#define __L1track2D_H__

#include "FWCore/Utilities/interface/Exception.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1trackBase.h"
#include "L1Trigger/TrackFindingTMTT/interface/Settings.h"
#include "L1Trigger/TrackFindingTMTT/interface/Utility.h"
#include "L1Trigger/TrackFindingTMTT/interface/TP.h"
#include "L1Trigger/TrackFindingTMTT/interface/Stub.h"

#include <vector>
#include <utility>

using namespace std;

//=== L1 track cand found in 2 dimensions.
//=== Gives access to all stubs on track and to its 2D helix parameters.
//=== Also calculates & gives access to associated truth particle (Tracking Particle) if any.

namespace TMTT {

class L1track2D : public L1trackBase {

public:

  // Give stubs on track, its cell location inside HT array, its 2D helix parameters. 
  L1track2D(const Settings* settings, const vector<const Stub*>& stubs, 
	    pair<unsigned int, unsigned int> cellLocationHT, pair<float, float> helix2D,
	    unsigned int iPhiSec, unsigned int iEtaReg, unsigned int optoLinkID, bool mergedHTcell) : 
            L1trackBase(),
	    settings_(settings),
	    stubs_(stubs), 
            cellLocationHT_(cellLocationHT),
            helix2D_(helix2D),
	    estValid_(false), estZ0_(0.), estTanLambda_(0.),
            iPhiSec_(iPhiSec), iEtaReg_(iEtaReg), optoLinkID_(optoLinkID), mergedHTcell_(mergedHTcell)
  {
    nLayers_   = Utility::countLayers(settings, stubs); // Count tracker layers these stubs are in
    matchedTP_ = Utility::matchingTP(settings, stubs, nMatchedLayers_, matchedStubs_); // Find associated truth particle & calculate info about match.
  }

  L1track2D() : L1trackBase() {}; // Creates track object, but doesn't set any variables.

  ~L1track2D() {}

  //--- Get information about the reconstructed track.

  // Get stubs on track candidate.
  const vector<const Stub*>&        getStubs()              const  {return stubs_;}  
  // Get number of stubs on track candidate.
  unsigned int                      getNumStubs()           const  {return stubs_.size();}
  // Get number of tracker layers these stubs are in.
  unsigned int                      getNumLayers()          const  {return nLayers_;}
  // Get cell location of track candidate in Hough Transform array in units of bin number.
  pair<unsigned int, unsigned int>  getCellLocationHT()     const  {return cellLocationHT_;}
  // The two conventionally agreed track helix parameters relevant in this 2D plane.
  // i.e. (q/Pt, phi0).
  pair<float, float>                getHelix2D()            const  {return helix2D_;}

  //--- User-friendlier access to the helix parameters obtained from track location inside HT array.

  float   qOverPt()    const  {return helix2D_.first;}
  float   phi0()       const  {return helix2D_.second;}

  //--- In the case of tracks found by the r-phi HT, a rough estimate of the (z0, tan_lambda) may be provided by any r-z
  //--- track filter run after the r-phi HT. These two functions give set/get access to these.
  //--- The "get" function returns a boolean indicating if an estimate exists (i.e. "set" has been called).

  void setTrkEstZ0andTanLam(float  estZ0, float  estTanLambda) {
    estZ0_ = estZ0; estTanLambda_ = estTanLambda; estValid_ = true;
  } 

  bool getTrkEstZ0andTanLam(float& estZ0, float& estTanLambda) const {
    estZ0 = estZ0_; estTanLambda = estTanLambda_; return estValid_;
  }

  //--- Get phi sector and eta region used by track finding code that this track is in.
  unsigned int iPhiSec() const  {return iPhiSec_;}
  unsigned int iEtaReg() const  {return iEtaReg_;}

  //--- Opto-link ID used to send this track from HT to Track Fitter. Both read & write functions.
  unsigned int optoLinkID() const {return optoLinkID_;}
  void setOptoLinkID(unsigned int linkID) {optoLinkID_ = linkID;}


  //--- Was this track produced from a marged HT cell (e.g. 2x2)?
  bool mergedHTcell() const {return mergedHTcell_;}

  //--- Get information about its association (if any) to a truth Tracking Particle.

  // Get matching tracking particle (=nullptr if none).
  const TP*                  getMatchedTP()          const   {return matchedTP_;}
  // Get the matched stubs.
  const vector<const Stub*>& getMatchedStubs()       const   {return matchedStubs_;}
  // Get number of matched stubs.
  unsigned int               getNumMatchedStubs()    const   {return matchedStubs_.size();}
  // Get number of tracker layers with matched stubs.
  unsigned int               getNumMatchedLayers()   const   {return nMatchedLayers_;}

private:

  //--- Configuration parameters
  const Settings*                    settings_; 

  //--- Information about the reconstructed track from Hough transform.
  vector<const Stub*>                stubs_;
  unsigned int                       nLayers_;
  pair<unsigned int, unsigned int>   cellLocationHT_; 
  pair<float, float>                 helix2D_; 

  //--- Rough estimate of r-z track parameters from r-z filter, which may be present in case of r-phi Hough transform
  bool  estValid_;
  float estZ0_;
  float estTanLambda_;

  unsigned int                       iPhiSec_;
  unsigned int                       iEtaReg_; 
  unsigned int                       optoLinkID_;

  bool                               mergedHTcell_;

  //--- Information about its association (if any) to a truth Tracking Particle.  
  const TP*                          matchedTP_;
  vector<const Stub*>                matchedStubs_;
  unsigned int                       nMatchedLayers_;
};

}

#endif
