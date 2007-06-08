#ifndef RecoMuon_MuonSeedGenerator_RPCSeedFinder_H
#define RecoMuon_MuonSeedGenerator_RPCSeedFinder_H

/** \class RPCSeedFinder
 *  
 *
 *  \author D. Pagano - University of Pavia & INFN Pavia
 *
 *  $Date: 2006/08/01 15:53:04 $
 *  $Revision: 1.1 $
 *  
 */

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"

#include <vector>

namespace edm {class EventSetup;}
class MagneticField;

class RPCSeedFinder {
public:
  
  RPCSeedFinder();

  virtual ~RPCSeedFinder(){};

  void add(MuonTransientTrackingRecHit::MuonRecHitPointer hit) { theRhits.push_back(hit); }
  
  std::vector<TrajectorySeed> seeds(const edm::EventSetup& eSetup) const;
  MuonTransientTrackingRecHit::ConstMuonRecHitPointer firstRecHit() const { return theRhits.front(); }
  unsigned int nrhit() const { return  theRhits.size(); }
  
private:
    
  MuonTransientTrackingRecHit::MuonRecHitContainer theRhits;
 
};
#endif
