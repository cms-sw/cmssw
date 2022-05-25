#ifndef RecoMuon_MuonSeedGenerator_RPCSeedFinder_H
#define RecoMuon_MuonSeedGenerator_RPCSeedFinder_H

/** \class RPCSeedFinder
 *  
 *   \author Haiyun.Teng - Peking University
 *
 *  
 */

#include "RecoMuon/MuonSeedGenerator/src/RPCSeedPattern.h"
#include <DataFormats/TrajectorySeed/interface/TrajectorySeed.h>
#include <RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <vector>
#include <algorithm>

class MagneticField;
class RPCGeometry;

class RPCSeedFinder {
  typedef MuonTransientTrackingRecHit::MuonRecHitPointer MuonRecHitPointer;
  typedef MuonTransientTrackingRecHit::ConstMuonRecHitPointer ConstMuonRecHitPointer;
  typedef MuonTransientTrackingRecHit::MuonRecHitContainer MuonRecHitContainer;
  typedef MuonTransientTrackingRecHit::ConstMuonRecHitContainer ConstMuonRecHitContainer;
  typedef RPCSeedPattern::weightedTrajectorySeed weightedTrajectorySeed;

public:
  RPCSeedFinder();
  ~RPCSeedFinder();
  void configure(const edm::ParameterSet &iConfig);
  void setOutput(std::vector<weightedTrajectorySeed> *goodweightedRef,
                 std::vector<weightedTrajectorySeed> *candidateweightedRef);
  void setrecHits(ConstMuonRecHitContainer &recHits);
  void setEventSetup(const MagneticField &field, const RPCGeometry &rpcGeom);
  void seed();

private:
  // Signal for call fillLayers()
  bool isrecHitsset;
  bool isConfigured;
  bool isOutputset;
  bool isEventSetupset;
  const MagneticField *pField = nullptr;
  const RPCGeometry *pRPCGeom = nullptr;
  RPCSeedPattern oneSeed;
  //ConstMuonRecHitContainer theRecHits;
  std::vector<weightedTrajectorySeed> *goodweightedSeedsRef;
  std::vector<weightedTrajectorySeed> *candidateweightedSeedsRef;
};
#endif
