#ifndef RecoMuon_MuonSeedGenerator_MuonDTSeedFromRecHits_H
#define RecoMuon_MuonSeedGenerator_MuonDTSeedFromRecHits_H

/**  \class MuonDTSeedFromRecHits
 *
 *  \author A. Vitelli - INFN Torino
 *  
 *  \author porting R.Bellan - INFN Torino
 *
 *   Generate a seed starting from a list of RecHits 
 *   make use of TrajectorySeed from CommonDet
 *
 */

#include "RecoMuon/MuonSeedGenerator/src/MuonSeedFromRecHits.h"

#include <vector>


namespace edm {class EventSetup;}

class MuonDTSeedFromRecHits : public MuonSeedFromRecHits
{
  public:
  MuonDTSeedFromRecHits(const edm::EventSetup& eSetup);

  virtual TrajectorySeed seed() const;

  private:
  MuonTransientTrackingRecHit::ConstMuonRecHitPointer best_cand() const;
  // was
  // TrackingRecHit best_cand() const;

  void computePtWithVtx(double* pt, double* spt) const;
  void computePtWithoutVtx(double* pt, double* spt) const;
  void computeBestPt(double* pt, double* spt, float& ptmean, float& sptmean) const;

};

#endif
