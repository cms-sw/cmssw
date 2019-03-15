#ifndef OMTF_GhostBusterPreferRefDt_H
#define OMTF_GhostBusterPreferRefDt_H

#include <L1Trigger/L1TMuonBayes/interface/Omtf/AlgoMuon.h>
#include <L1Trigger/L1TMuonBayes/interface/Omtf/IGhostBuster.h>
#include <L1Trigger/L1TMuonBayes/interface/Omtf/OMTFConfiguration.h>
#include <vector>
#include <ostream>

#include <map>
#include <set>

#include <memory>


class GhostBusterPreferRefDt: public IGhostBuster {
private:
  const OMTFConfiguration* omtfConfig;
public:
  GhostBusterPreferRefDt(const OMTFConfiguration* omtfConfig):omtfConfig(omtfConfig) {};

  ~GhostBusterPreferRefDt() override {};

  AlgoMuons select(AlgoMuons refHitCands, int charge=0) override;

};
#endif
