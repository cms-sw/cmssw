#ifndef RecoMuon_MuonSeedGenerator_MuonSeedFinder_H
#define RecoMuon_MuonSeedGenerator_MuonSeedFinder_H

/** \class MuonSeedFinder
 *  No description available.
 *
 *  \author A. Vitelli - INFN Torino
 *
 *  \author porting R. Bellan - INFN Torino
 *
 *  $Date: $
 *  $Revision: $
 *  
 */

// FIXME!! It's dummy
#include "DataFormats/TrackReco/interface/RecHit.h" 
//was
//#include "CommonDet/BasicDet/interface/RecHit.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
// was
//#include "CommonDet/PatternPrimitives/interface/TrajectoryStateOnSurface.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
// was a vector of
//#include "Muon/MuonSeedGenerator/interface/MuonTrajectorySeed.h"

#include <vector>

//class RecHit;
//class BoundPlane;


class MuonSeedFinder {

  //FIXME
public:
  typedef std::vector<RecHit>               RecHitContainer;
  typedef RecHitContainer::const_iterator   RecHitIterator;

public:
  /// Constructor
  MuonSeedFinder();

  /// Destructor
  virtual ~MuonSeedFinder();

  // Operations

  void add(const RecHit& hit) { theRhits.push_back(hit); }
  
  TrajectorySeedCollection seeds() const;
  const RecHit& rhit() const { return theRhits.front(); }
  unsigned int nrhit() const { return  theRhits.size(); }
  
private:
  RecHit best_cand(RecHit* rhit=0) const;
  bool createEndcapSeed(RecHit me, TrajectorySeedCollection& theSeeds) const;
  
  RecHitContainer theRhits;
 
  // put a parameterSet instead of
  // static SimpleConfigurable<float> theMinMomentum;
  float theMinMomentum;
};
#endif
