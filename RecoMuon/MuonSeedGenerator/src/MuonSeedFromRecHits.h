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

#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include <vector>


namespace edm {class EventSetup;}

class MuonSeedFromRecHits 
{
public:
  MuonSeedFromRecHits(const edm::EventSetup & eSetup);
  virtual ~MuonSeedFromRecHits() {}

  void add(MuonTransientTrackingRecHit::MuonRecHitPointer hit) { theRhits.push_back(hit); }
  MuonTransientTrackingRecHit::ConstMuonRecHitPointer firstRecHit() const { return theRhits.front(); }
  unsigned int nrhit() const { return  theRhits.size(); }

  TrajectorySeed createSeed(float ptmean, float sptmean,
			    MuonTransientTrackingRecHit::ConstMuonRecHitPointer last) const;
  
  protected:
  friend class MuonSeedFinder;
  typedef MuonTransientTrackingRecHit::MuonRecHitContainer MuonRecHitContainer;
  typedef MuonTransientTrackingRecHit::MuonRecHitPointer MuonRecHitPointer;
  typedef MuonTransientTrackingRecHit::ConstMuonRecHitPointer ConstMuonRecHitPointer;

  // makes a straight-line seed.  q/p = 0, and sigma(q/p) = 1/theMinMomentum
  TrajectorySeed createDefaultSeed(MuonTransientTrackingRecHit::ConstMuonRecHitPointer last) const;

  protected:
  MuonTransientTrackingRecHit::MuonRecHitContainer theRhits;
  const MagneticField * theField;

};

#endif
