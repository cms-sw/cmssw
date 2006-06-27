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

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"

#include <vector>

class MuonTransientTrackingRecHit;

class RecHit;
class BoundPlane;

namespace edm {class EventSetup;}

class MuonSeedFromRecHits {
  typedef std::vector<MuonTransientTrackingRecHit*>  RecHitContainer;
  typedef RecHitContainer::const_iterator            RecHitIterator;

  public:
  MuonSeedFromRecHits(){
  }

  void add(MuonTransientTrackingRecHit* hit) { theRhits.push_back(hit); }
  TrajectorySeed seed(const edm::EventSetup& eSetup) const;
  const MuonTransientTrackingRecHit* firstRecHit() const { return theRhits.front(); }
  unsigned int nrhit() const { return  theRhits.size(); }

  private:
  MuonTransientTrackingRecHit *best_cand() const;
  // was
  // TrackingRecHit best_cand() const;

  void computePtWithVtx(double* pt, double* spt) const;
  void computePtWithoutVtx(double* pt, double* spt) const;
  void computeBestPt(double* pt, double* spt, float& ptmean, float& sptmean) const;

  TrajectorySeed createSeed(float ptmean, float sptmean,const edm::EventSetup& eSetup) const;

  private:
  RecHitContainer theRhits;
};

#endif
