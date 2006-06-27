#ifndef RecoMuon_MuonSeedGenerator_MuonSeedFinder_H
#define RecoMuon_MuonSeedGenerator_MuonSeedFinder_H

/** \class MuonSeedFinder
 *  
 *  Uses SteppingHelixPropagator
 *
 *  \author A. Vitelli - INFN Torino
 *  \author porting R. Bellan - INFN Torino
 *
 *  $Date: 2006/05/30 13:50:23 $
 *  $Revision: 1.5 $
 *  
 */

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/CSCRecHit/interface/CSCRecHit2D.h"

#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"

#include <vector>

namespace edm {class EventSetup;}
class MagneticField;

class MuonSeedFinder {

public:
  
  typedef std::vector<MuonTransientTrackingRecHit*>       RecHitContainer;
  typedef RecHitContainer::const_iterator   RecHitIterator;

public:
  /// Constructor
  MuonSeedFinder();

  /// Destructor
  virtual ~MuonSeedFinder(){};

  // Operations

  void add(MuonTransientTrackingRecHit* hit) { theRhits.push_back(hit); }
  
  std::vector<TrajectorySeed> seeds(const edm::EventSetup& eSetup) const;
  MuonTransientTrackingRecHit *firstRecHit() const { return theRhits.front(); }
  unsigned int nrhit() const { return  theRhits.size(); }
  
private:
  //  TrackingRecHit best_cand(TrackingRecHit* rhit=0) const;
  bool createEndcapSeed(MuonTransientTrackingRecHit *me, 
			std::vector<TrajectorySeed>& theSeeds,
			const edm::EventSetup& eSetup) const;
  
  float computePt(const MuonTransientTrackingRecHit *muon, const MagneticField *field) const;

  RecHitContainer theRhits;
 
  // put a parameterSet instead of
  // static SimpleConfigurable<float> theMinMomentum;
  float theMinMomentum;
};
#endif
