#ifndef RecoMuon_MuonSeedGenerator_MuonSeedVFinder_H
#define RecoMuon_MuonSeedGenerator_MuonSeedVFinder_H

/** \class MuonSeedFinder
 *  
 *  Uses SteppingHelixPropagator
 *
 *  \author R. Wilkinson
 *
 *  $Date: 2008/10/17 23:17:08 $
 *  $Revision: 1.2 $
 *  
 */

#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "RecoMuon/MuonSeedGenerator/src/MuonSeedPtExtractor.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include <vector>

class MuonSeedVFinder {
public:

  virtual ~MuonSeedVFinder() {}
  virtual void setBField(const MagneticField * field) = 0;

  virtual void seeds(const MuonTransientTrackingRecHit::MuonRecHitContainer & hits, 
                     std::vector<TrajectorySeed> & result) = 0;
  
  void setBeamSpot(const GlobalVector & gv) {thePtExtractor->setBeamSpot(gv);}
protected:
  MuonSeedPtExtractor * thePtExtractor;

};
#endif
