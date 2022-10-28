/*
 * IGhostBuster.h
 *
 *  Created on: Jun 28, 2017
 *      Author: kbunkow
 */

#ifndef L1T_OmtfP1_IGHOSTBUSTER_H_
#define L1T_OmtfP1_IGHOSTBUSTER_H_

#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/AlgoMuon.h"

class IGhostBuster {
public:
  virtual ~IGhostBuster() {}

  virtual AlgoMuons select(AlgoMuons refHitCands, int charge = 0) = 0;
};

#endif /* L1T_OmtfP1_IGHOSTBUSTER_H_ */
