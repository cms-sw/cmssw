#ifndef DataFormats_MuonSeed_L3MuonTrajectorySeed_H
#define DataFormats_MuonSeed_L3MuonTrajectorySeed_H

/** \class L3MuonTrajectorySeed
 *  Concrete class for the seed used by the second level of the muon HLT.
 *  It stores the information (and the link) from the L1 particle 
 *
 *  $Date: 2008/02/13 14:49:28 $
 *  $Revision: 1.1 $
 *  \author  J.-R. Vlimant 
 */

#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

class L3MuonTrajectorySeed: public TrajectorySeed {
public:
  typedef edm::OwnVector<TrackingRecHit> RecHitContainer;

  /// Default constructor
  L3MuonTrajectorySeed() {}
  
  /// Constructor with L1 ref
  L3MuonTrajectorySeed(const TrajectorySeed & base,
		       const l1extra::L1MuonParticleRef & l1Ref) :
    TrajectorySeed(base), theL1Particle(l1Ref){}

  /// Constructor with L2 ref
  L3MuonTrajectorySeed(const TrajectorySeed & base,
		       const reco::TrackRef & l2Ref) :
    TrajectorySeed(base), theL2Track(l2Ref){}

  /// Destructor
  virtual ~L3MuonTrajectorySeed(){};

  //accessors

  /// Get L1 info
  inline l1extra::L1MuonParticleRef l1Particle() const {return theL1Particle;}

  /// Get L2 info
    inline reco::TrackRef l2Track() const { return theL2Track;}

protected:

private:
  l1extra::L1MuonParticleRef theL1Particle;
  reco::TrackRef theL2Track;
};
#endif

