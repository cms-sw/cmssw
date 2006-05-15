#ifndef RecoMuon_MuonSeedGenerator_MuonSeedFinder_H
#define RecoMuon_MuonSeedGenerator_MuonSeedFinder_H

/** \class MuonSeedFinder
 *  No description available.
 *
 *  \author A. Vitelli - INFN Torino
 *
 *  \author porting R. Bellan - INFN Torino
 *
 *  $Date: 2006/03/24 11:43:48 $
 *  $Revision: 1.1 $
 *  
 */

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
// was
//#include "CommonDet/PatternPrimitives/interface/TrajectoryStateOnSurface.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
// was a vector of
//#include "Muon/MuonSeedGenerator/interface/MuonTrajectorySeed.h"

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/CSCRecHit/interface/CSCRecHit2D.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

#include <vector>

class MuonSeedFinder {

  //FIXME
public:
  
  typedef std::vector<TransientTrackingRecHit*>       RecHitContainer;
  typedef RecHitContainer::const_iterator   RecHitIterator;

public:
  /// Constructor
  MuonSeedFinder();

  /// Destructor
  virtual ~MuonSeedFinder(){};

  // Operations

  void add(TransientTrackingRecHit* hit) { theRhits.push_back(hit); }
  
  std::vector<TrajectorySeed> seeds() const;
  TransientTrackingRecHit *firstRecHit() const { return theRhits.front(); }
  unsigned int nrhit() const { return  theRhits.size(); }
  
private:
  //  TrackingRecHit best_cand(TrackingRecHit* rhit=0) const;
  bool createEndcapSeed(TransientTrackingRecHit *me, std::vector<TrajectorySeed>& theSeeds) const;
  
  RecHitContainer theRhits;
 
  // put a parameterSet instead of
  // static SimpleConfigurable<float> theMinMomentum;
  float theMinMomentum;
  bool debug;
};
#endif
