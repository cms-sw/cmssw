#ifndef DataFormats_MuonSeed_L2MuonTrajectorySeed_H
#define DataFormats_MuonSeed_L2MuonTrajectorySeed_H

/** \class L2MuonTrajectorySeed
 *  Concrete class for the seed used by the second level of the muon HLT.
 *  It stores the information (and the link) from the L1 particle 
 *
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/L1TMuonPhase2/interface/TrackerMuon.h"
#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"

class L2MuonTrajectorySeed : public TrajectorySeed {
public:
  /// Default constructor
  L2MuonTrajectorySeed();

  /// Constructor
  L2MuonTrajectorySeed(PTrajectoryStateOnDet const& ptsos,
                       RecHitContainer const& rh,
                       PropagationDirection dir,
                       l1extra::L1MuonParticleRef l1Ref);

  /// Constructor for stage2 L1
  L2MuonTrajectorySeed(PTrajectoryStateOnDet const& ptsos,
                       RecHitContainer const& rh,
                       PropagationDirection dir,
                       l1t::MuonRef l1Ref);

  L2MuonTrajectorySeed(PTrajectoryStateOnDet const& ptsos,
                       RecHitContainer const& rh,
                       PropagationDirection dir,
                       l1t::TrackerMuonRef l1TkMuRef);

  /// Destructor
  ~L2MuonTrajectorySeed() override {}

  // Operations

  /// Get L1 info
  inline l1extra::L1MuonParticleRef l1Particle() const { return theL1Particle; }
  inline l1t::MuonRef l1tParticle() const { return theL1TParticle; }
  inline l1t::TrackerMuonRef l1TkMu() const { return theL1TkMu; }

private:
  l1extra::L1MuonParticleRef theL1Particle;
  l1t::MuonRef theL1TParticle;
  l1t::TrackerMuonRef theL1TkMu;
};
#endif
