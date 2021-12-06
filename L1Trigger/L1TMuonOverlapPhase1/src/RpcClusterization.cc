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

  for (auto& digi : digis) {
    if (allClusters.empty()) {
      allClusters.emplace_back(digi.strip(), digi.strip());
      allClusters.back().bx = digi.bx();
      allClusters.back().timing = convertTiming(digi.time());
    } else if (digi.strip() - allClusters.back().lastStrip == 1) {
      allClusters.back().lastStrip = digi.strip();
      //TODO update bx and timing in some smart way
    } else if (digi.strip() - allClusters.back().lastStrip > 1) {
      allClusters.emplace_back(digi.strip(), digi.strip());
      allClusters.back().bx = digi.bx();
      allClusters.back().timing = convertTiming(digi.time());
    }
  }

  std::vector<RpcCluster> filteredClusters;

  if (dropAllClustersIfMoreThanMax)
    if (allClusters.size() > maxClusterCnt)
      return filteredClusters;

  //debug printout only
  if (allClusters.size() > maxClusterCnt) {
    LogTrace("l1tOmtfEventPrint") << __FUNCTION__ << ":" << __LINE__ << " allClusters.size() >= maxClusterCnt "
                                  << std::endl;
    for (auto& cluster : allClusters)
      edm::LogVerbatim("l1tOmtfEventPrint")
          << __FUNCTION__ << ":" << __LINE__ << " roll " << roll << " cluster: firstStrip " << cluster.firstStrip
          << " lastStrip " << cluster.lastStrip << " halfStrip " << cluster.halfStrip() << std::endl;
  }

  //TODO this is very simple filtering of the cluster,
  //unfortunately the in firmware it is more complicated and cannot be easily emulated from digi
  //(in principle would required raws, because in the firmware the clusterizaton is based on the 8-bit strip partitions
  for (auto& cluster : allClusters) {
    if (cluster.size() <= maxClusterSize)
      filteredClusters.emplace_back(cluster);

    if (filteredClusters.size() >= maxClusterCnt)
      break;
  }

  return filteredClusters;
}

int RpcClusterization::convertTiming(double timing) const {
  return timing;  //TODO implement
}
