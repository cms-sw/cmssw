#ifndef OMTF_GhostBusterPreferRefDt_H
#define OMTF_GhostBusterPreferRefDt_H

#include <vector>
#include <ostream>

#include <map>
#include <set>

#include <memory>

#include "L1Trigger/L1TMuonOverlap/interface/IGhostBuster.h"
#include "L1Trigger/L1TMuonOverlap/interface/AlgoMuon.h"
#include "L1Trigger/L1TMuonOverlap/interface/OMTFConfiguration.h"

class GhostBusterPreferRefDt: public IGhostBuster {
private:
  const OMTFConfiguration* omtfConfig;
public:
  GhostBusterPreferRefDt(OMTFConfiguration* omtfConfig):omtfConfig(omtfConfig) {};

  virtual ~GhostBusterPreferRefDt() {};

  virtual std::vector<AlgoMuon> select(std::vector<AlgoMuon> refHitCands, int charge=0);

};
#endif
