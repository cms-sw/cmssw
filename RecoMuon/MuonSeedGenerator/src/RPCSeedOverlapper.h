#ifndef RecoMuon_MuonSeedGenerator_RPCSeedOverlapper_H
#define RecoMuon_MuonSeedGenerator_RPCSeedOverlapper_H

/**  \class RPCSeedPattern
 *
 *  \author Haiyun.Teng - Peking University
 *
 *
 */

#include <DataFormats/TrajectorySeed/interface/TrajectorySeed.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <Geometry/RPCGeometry/interface/RPCGeometry.h>
#include <DataFormats/Common/interface/OwnVector.h>
#include <DataFormats/TrackingRecHit/interface/TrackingRecHit.h>
#include "RecoMuon/MuonSeedGenerator/src/RPCSeedPattern.h"

class RPCSeedOverlapper {
  typedef RPCSeedPattern::weightedTrajectorySeed weightedTrajectorySeed;

public:
  RPCSeedOverlapper();
  ~RPCSeedOverlapper();
  void setIO(std::vector<weightedTrajectorySeed> *goodweightedRef,
             std::vector<weightedTrajectorySeed> *candidateweightedRef);
  void unsetIO();
  void run();
  void configure(const edm::ParameterSet &iConfig);
  void setGeometry(const RPCGeometry &iGeom);

private:
  void CheckOverlap(const RPCGeometry &iGeom, std::vector<weightedTrajectorySeed> *SeedsRef);
  bool isShareHit(const std::vector<TrackingRecHit const *> &RecHits,
                  const TrackingRecHit &hit,
                  const RPCGeometry &rpcGeometry);
  // Signal for call run()
  bool isConfigured;
  bool isIOset;
  // Parameters for configuration
  bool isCheckgoodOverlap;
  bool isCheckcandidateOverlap;
  unsigned int ShareRecHitsNumberThreshold;
  // IO ref
  std::vector<weightedTrajectorySeed> *goodweightedSeedsRef;
  std::vector<weightedTrajectorySeed> *candidateweightedSeedsRef;
  const RPCGeometry *rpcGeometry;
};

#endif
