/*
 * IGhostBuster.h
 *
 *  Created on: Jun 28, 2017
 *      Author: kbunkow
 */

#ifndef OMTF_IGHOSTBUSTER_H_
#define OMTF_IGHOSTBUSTER_H_

#include "L1Trigger/L1TMuonOverlap/interface/AlgoMuon.h"

class IGhostBuster {
public:
  virtual ~IGhostBuster() {}

  virtual std::vector<AlgoMuon> select(std::vector<AlgoMuon> refHitCands, int charge=0) = 0;

};

#endif /* OMTF_IGHOSTBUSTER_H_ */
