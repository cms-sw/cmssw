/*
 * RpcClusterization.cpp
 *
 *  Created on: Jan 14, 2019
 *      Author: kbunkow
 */

#include "L1Trigger/L1TMuonOverlapPhase1/interface/RpcClusterization.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <cmath>
#include <algorithm>

RpcClusterization::~RpcClusterization() {}

std::vector<RpcCluster> RpcClusterization::getClusters(const RPCDetId& roll, std::vector<RPCDigi>& digis) const {
  std::vector<RpcCluster> allClusters;

  std::sort(digis.begin(), digis.end(), [](const RPCDigi& a, const RPCDigi& b) { return a.strip() < b.strip(); });

  typedef std::pair<unsigned int, unsigned int> Cluster;

  //This implementation of clusterization emulation gives the cluster in the same order as the order of digis,
  //and the order of unpacked digis should be the same as the order of the LB channels on which the clustrization
  //in the firmware is performed.
  //This cluster order plays role in some rare cases for the OMTF algorithm
  //when two hits has the same abs(minDistPhi), and then the phi of the resulting candidate
  //depends on the order of these hits.
  for (unsigned int iDigi = 0; iDigi < digis.size(); iDigi++) {
    //edm::LogVerbatim("l1tOmtfEventPrint")<< __FUNCTION__ << ":" << __LINE__<<" "<<roll<<" iDigi "<<iDigi<<" digi "<<digis[iDigi];

    //removing duplicated digis
    //the digis might be duplicated, because the same data might be received by two OMTF boards (as the same link goes to two neighboring boards)
    //and the unpacker is not cleaning them
    bool duplicatedDigi = false;
    for (unsigned int iDigi2 = 0; iDigi2 < iDigi; iDigi2++) {
      if (digis[iDigi].strip() == digis[iDigi2].strip()) {
        duplicatedDigi = true;
        //edm::LogVerbatim("l1tOmtfEventPrint")<<"duplicatedDigi";
        break;
      }
    }

    if (duplicatedDigi)
      continue;

    bool addNewCluster = true;

    for (auto& cluster : allClusters) {
      if (digis[iDigi].strip() - cluster.lastStrip == 1) {
        cluster.lastStrip = digis[iDigi].strip();
        addNewCluster = false;
      } else if (digis[iDigi].strip() - cluster.firstStrip == -1) {
        cluster.firstStrip = digis[iDigi].strip();
        addNewCluster = false;
      } else if (digis[iDigi].strip() >= cluster.firstStrip && digis[iDigi].strip() <= cluster.lastStrip) {
        addNewCluster = false;
      }
    }

    if (addNewCluster) {
      allClusters.emplace_back(digis[iDigi].strip(), digis[iDigi].strip());
      allClusters.back().bx = digis[iDigi].bx();
      allClusters.back().timing = convertTiming(digis[iDigi].time());
    }
  }

  /* Debug only
  if(allClusters.size())
	  edm::LogVerbatim("l1tOmtfEventPrint")<< __FUNCTION__ <<" "<<roll<<" allClusters.size() "<<allClusters.size();
  for (auto& cluster : allClusters)
	  edm::LogVerbatim("l1tOmtfEventPrint")
        << __FUNCTION__ << " cluster: firstStrip " << cluster.firstStrip
        << " lastStrip " << cluster.lastStrip << " halfStrip " << cluster.halfStrip() << std::endl;*/

  return allClusters;
}

int RpcClusterization::convertTiming(double timing) const {
  return timing;  //TODO implement
}
