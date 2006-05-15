#ifndef RecoMuon_MuonSeedGenerator_MuonSeedFromRecHits_H
#define RecoMuon_MuonSeedGenerator_MuonSeedFromRecHits_H

/**  \class MuonSeedFromRecHits
 *
 *  \author A. Vitelli - INFN Torino
 *  
 *  \author porting R.Bellan - INFN Torino
 *
 *   Generate a seed starting from a list of RecHits 
 *   make use of TrajectorySeed from CommonDet
 *
 */

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
// was
// #include "CommonDet/PatternPrimitives/interface/TrajectoryStateOnSurface.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"

#include <vector>

class TransientTrackingRecHit;

class RecHit;
class BoundPlane;

class MuonSeedFromRecHits {
  typedef std::vector<TransientTrackingRecHit*>       RecHitContainer;
  typedef RecHitContainer::const_iterator   RecHitIterator;

  public:
  MuonSeedFromRecHits(){
    debug = true;
  }

  void add(TransientTrackingRecHit* hit) { theRhits.push_back(hit); }
  TrajectorySeed seed() const;
  const TransientTrackingRecHit* firstRecHit() const { return theRhits.front(); }
  unsigned int nrhit() const { return  theRhits.size(); }

  private:
  TransientTrackingRecHit *best_cand() const;
  // was
  // TrackingRecHit best_cand() const;

  void computePtWithVtx(double* pt, double* spt) const;
  void computePtWithoutVtx(double* pt, double* spt) const;
  void computeBestPt(double* pt, double* spt, float& ptmean, float& sptmean) const;

  TrajectorySeed createSeed(float ptmean, float sptmean) const;

  private:
  RecHitContainer theRhits;
  bool debug;
};

#endif
