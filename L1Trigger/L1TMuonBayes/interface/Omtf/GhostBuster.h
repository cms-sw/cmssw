#ifndef OMTF_GhostBuster_H
#define OMTF_GhostBuster_H

#include <L1Trigger/L1TMuonBayes/interface/Omtf/AlgoMuon.h>
#include <L1Trigger/L1TMuonBayes/interface/Omtf/IGhostBuster.h>
#include <vector>
#include <ostream>

#include <map>
#include <set>

#include <memory>


class GhostBuster: public IGhostBuster {
public:
  ~GhostBuster() override {};
  AlgoMuons select(AlgoMuons refHitCands, int charge=0) override;

};
#endif
