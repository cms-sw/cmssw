#ifndef OMTF_GhostBuster_H
#define OMTF_GhostBuster_H

#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/AlgoMuon.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/IGhostBuster.h"

#include <vector>
#include <ostream>

#include <map>
#include <set>

#include <memory>

class GhostBuster : public IGhostBuster {
private:
  const OMTFConfiguration* omtfConfig;

public:
  GhostBuster(const OMTFConfiguration* omtfConfig) : omtfConfig(omtfConfig){};

  ~GhostBuster() override{};
  AlgoMuons select(AlgoMuons refHitCands, int charge = 0) override;
};
#endif
