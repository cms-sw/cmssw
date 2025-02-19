#ifndef DataFormats_MuonSeed_L2MuonTrajectorySeed_H
#define DataFormats_MuonSeed_L2MuonTrajectorySeed_H

/** \class L2MuonTrajectorySeed
 *  Concrete class for the seed used by the second level of the muon HLT.
 *  It stores the information (and the link) from the L1 particle 
 *
 *  $Date: 2007/12/17 14:51:19 $
 *  $Revision: 1.1 $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"

class L2MuonTrajectorySeed: public TrajectorySeed {
public:
  typedef edm::OwnVector<TrackingRecHit> RecHitContainer;

  /// Default constructor
  L2MuonTrajectorySeed();
  
  /// Constructor
  L2MuonTrajectorySeed(PTrajectoryStateOnDet const & ptsos, 
		       RecHitContainer const & rh, 
		       PropagationDirection  dir,
		       l1extra::L1MuonParticleRef l1Ref);

  /// Destructor
  virtual ~L2MuonTrajectorySeed(){};

  // Operations

  /// Get L1 info
  inline l1extra::L1MuonParticleRef l1Particle() const {return theL1Particle;}

protected:

private:
  l1extra::L1MuonParticleRef theL1Particle;
};
#endif

