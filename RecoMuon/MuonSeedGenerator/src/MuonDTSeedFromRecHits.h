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

#include "RecoMuon/TrackingTools/interface/MuonSeedFromRecHits.h"

#include <vector>


class MuonDTSeedFromRecHits : public MuonSeedFromRecHits
{
  public:
  MuonDTSeedFromRecHits();

  virtual TrajectorySeed seed() const;

  ConstMuonRecHitPointer bestBarrelHit(const MuonRecHitContainer & barrelHits) const;
  // was
  // TrackingRecHit best_cand() const;

private:
  void computePtWithVtx(double* pt, double* spt) const;
  void computePtWithoutVtx(double* pt, double* spt) const;
  void computeBestPt(double* pt, double* spt, float& ptmean, float& sptmean) const;

  // picks the segment that's nearest in eta to the most other segments
  float bestEta() const;
  void computeMean(const double* pt, const double * weights, int sz,
                   bool tossOutlyers, float& ptmean, float & sptmean) const;

};

#endif
