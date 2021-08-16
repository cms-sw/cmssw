/*
 * RpcClusterization.h
 *
 *  Created on: Jan 14, 2019
 *      Author: kbunkow
 */

#ifndef INTERFACE_RPCCLUSTERIZATION_H_
#define INTERFACE_RPCCLUSTERIZATION_H_

#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"

#include <vector>

class RpcCluster {
public:
  int firstStrip = -1;
  int lastStrip = -1;

  int bx = 0;
  int timing = 0;  //sub-bx timing, should be already in scale common for all muon subsystems

  RpcCluster(unsigned int firstStrip, unsigned int lastStrip) : firstStrip(firstStrip), lastStrip(lastStrip){};

  float halfStrip() { return (lastStrip + firstStrip) / 2.; }

  unsigned int size() const { return abs(firstStrip - lastStrip) + 1; }
};

class RpcClusterization {
public:
  RpcClusterization(){};

  virtual ~RpcClusterization();

  void configure(int maxClusterSize, int maxClusterCnt, bool dropAllClustersIfMoreThanMax) {
    this->maxClusterSize = maxClusterSize;
    this->maxClusterCnt = maxClusterCnt;
    this->dropAllClustersIfMoreThanMax = dropAllClustersIfMoreThanMax;
  }

  ///N.B. digis are sorted inside the function
  virtual std::vector<RpcCluster> getClusters(const RPCDetId& roll, std::vector<RPCDigi>& digis) const;

  //converts float timing to the int timing in the scale common for the muon detectors
  virtual int convertTiming(double timing) const;

private:
  unsigned int maxClusterSize = 3;
  unsigned int maxClusterCnt = 2;

  bool dropAllClustersIfMoreThanMax =
      true;  // if true no  cluster is return if there is more clusters then maxClusterCnt (counted regardless of the size)
};

#endif /* INTERFACE_RPCCLUSTERIZATION_H_ */
