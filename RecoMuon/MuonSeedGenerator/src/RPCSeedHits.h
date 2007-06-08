#ifndef RecoMuon_MuonSeedGenerator_RPCSeedHits_H
#define RecoMuon_MuonSeedGenerator_RPCSeedHits_H

/**  \class RPCSeedHits
 *
 *  \author D. Pagano - University of Pavia & INFN Pavia
 *
 *
 */

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "TFile.h"
#include "TH1F.h"
#include <vector>

class RecHit;
class BoundPlane;
class GeomDet;

namespace edm {class EventSetup;}

class RPCSeedHits {
  typedef std::pair<const GeomDet*,TrajectoryStateOnSurface> DetWithState;

  public:
  RPCSeedHits(){}

  void add(MuonTransientTrackingRecHit::MuonRecHitPointer hit) { theRhits.push_back(hit); }
  TrajectorySeed seed(const edm::EventSetup& eSetup) const;
  MuonTransientTrackingRecHit::ConstMuonRecHitPointer firstRecHit() const { return theRhits.front(); }
  unsigned int nrhit() const { return  theRhits.size(); }

  private:
  friend class RPCSeedFinder;

  MuonTransientTrackingRecHit::ConstMuonRecHitPointer best_cand() const;

  void computePtWithoutVtx(double* pt, double* spt) const;
  void computeBestPt(double* pt, double* spt, float& ptmean, float& sptmean) const;

  TrajectorySeed createSeed(float ptmean, float sptmean,
			    MuonTransientTrackingRecHit::ConstMuonRecHitPointer last,
			    const edm::EventSetup& eSetup) const;

  private:

  MuonTransientTrackingRecHit::MuonRecHitContainer theRhits;
};

#endif
