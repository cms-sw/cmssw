#ifndef OMTF_GhostBuster_H
#define OMTF_GhostBuster_H

#include <vector>
#include <ostream>

#include <map>
#include <set>

#include <memory>

#include "L1Trigger/L1TMuonOverlap/interface/AlgoMuon.h"

class OMTFGhostBuster {

  public:
  
    void select(std::vector<AlgoMuon> & refHitCands, int charge=0);   

};
#endif