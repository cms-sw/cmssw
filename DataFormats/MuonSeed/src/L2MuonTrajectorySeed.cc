/** \class L2MuonTrajectorySeed
 *  Concrete class for the seed used by the second level of the muon HLT.
 *  It stores the information (and the link) from the L1 particle 
 *
 *  $Date: 2007/12/17 14:51:20 $
 *  $Revision: 1.1 $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "DataFormats/MuonSeed/interface/L2MuonTrajectorySeed.h"

// Default constructor
L2MuonTrajectorySeed::L2MuonTrajectorySeed():TrajectorySeed(){}

// Constructor
L2MuonTrajectorySeed::L2MuonTrajectorySeed(PTrajectoryStateOnDet const & ptsos, 
					   recHitContainer const & rh, 
					   PropagationDirection  dir,
					   l1extra::L1MuonParticleRef l1Ref):TrajectorySeed(ptsos, rh, dir){
  theL1Particle = l1Ref;
}

