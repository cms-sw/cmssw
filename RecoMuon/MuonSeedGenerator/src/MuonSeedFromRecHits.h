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
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"

#include <vector>

class RecHit;
class BoundPlane;
class GeomDet;

namespace edm {class EventSetup;}

class MuonSeedFromRecHits {
  typedef std::pair<const GeomDet*,TrajectoryStateOnSurface> DetWithState;

  public:
  MuonSeedFromRecHits(){}

  void add(MuonTransientTrackingRecHit::MuonRecHitPointer hit) { theRhits.push_back(hit); }
  TrajectorySeed seed(const edm::EventSetup& eSetup) const;
  MuonTransientTrackingRecHit::ConstMuonRecHitPointer firstRecHit() const { return theRhits.front(); }
  unsigned int nrhit() const { return  theRhits.size(); }

  private:
  friend class MuonSeedFinder;

  MuonTransientTrackingRecHit::ConstMuonRecHitPointer best_cand() const;
  // was
  // TrackingRecHit best_cand() const;

  void computePtWithVtx(double* pt, double* spt) const;
  void computePtWithoutVtx(double* pt, double* spt) const;
  void computeBestPt(double* pt, double* spt, float& ptmean, float& sptmean) const;

  TrajectorySeed createSeed(float ptmean, float sptmean,
			    MuonTransientTrackingRecHit::ConstMuonRecHitPointer last,
			    const edm::EventSetup& eSetup) const;

  private:
  MuonTransientTrackingRecHit::MuonRecHitContainer theRhits;
};

#endif
