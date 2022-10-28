/*
 * ProcConfigurationBase.cc
 *
 *  Created on: Jan 30, 2019
 *      Author: Karol Bunkowski kbunkow@cern.ch
 */

#include "L1Trigger/L1TMuonOverlapPhase1/interface/ProcConfigurationBase.h"
#include "DataFormats/CSCDigi/interface/CSCConstants.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

ProcConfigurationBase::ProcConfigurationBase() : cscLctCentralBx_(CSCConstants::LCT_CENTRAL_BX) {}

ProcConfigurationBase::~ProcConfigurationBase() {}

int ProcConfigurationBase::foldPhi(int phi) const {
  int phiBins = nPhiBins();
  if (phi > phiBins / 2)
    return (phi - phiBins);
  else if (phi < -phiBins / 2)
    return (phi + phiBins);

  return phi;
}

void ProcConfigurationBase::configureFromEdmParameterSet(const edm::ParameterSet& edmParameterSet) {
  if (edmParameterSet.exists("rpcMaxClusterSize")) {
    setRpcMaxClusterSize(edmParameterSet.getParameter<int>("rpcMaxClusterSize"));
    edm::LogVerbatim("OMTFReconstruction")
        << "rpcMaxClusterSize: " << edmParameterSet.getParameter<int>("rpcMaxClusterSize") << std::endl;
  }

  if (edmParameterSet.exists("rpcMaxClusterCnt")) {
    setRpcMaxClusterCnt(edmParameterSet.getParameter<int>("rpcMaxClusterCnt"));
    edm::LogVerbatim("OMTFReconstruction")
        << "rpcMaxClusterCnt: " << edmParameterSet.getParameter<int>("rpcMaxClusterCnt") << std::endl;
  }

  if (edmParameterSet.exists("rpcDropAllClustersIfMoreThanMax")) {
    setRpcDropAllClustersIfMoreThanMax(edmParameterSet.getParameter<bool>("rpcDropAllClustersIfMoreThanMax"));
    edm::LogVerbatim("OMTFReconstruction")
        << "rpcDropAllClustersIfMoreThanMax: " << edmParameterSet.getParameter<bool>("rpcDropAllClustersIfMoreThanMax")
        << std::endl;
  }

  if (edmParameterSet.exists("lctCentralBx")) {
    cscLctCentralBx_ = edmParameterSet.getParameter<int>("lctCentralBx");
    edm::LogVerbatim("OMTFReconstruction")
        << "lctCentralBx: " << edmParameterSet.getParameter<int>("lctCentralBx") << std::endl;
  }

  if (edmParameterSet.exists("minDtPhiQuality")) {
    minDtPhiQuality = edmParameterSet.getParameter<int>("minDtPhiQuality");
    edm::LogVerbatim("OMTFReconstruction")
        << "minDtPhiQuality: " << edmParameterSet.getParameter<int>("minDtPhiQuality") << std::endl;
  }

  if (edmParameterSet.exists("minDtPhiBQuality")) {
    minDtPhiBQuality = edmParameterSet.getParameter<int>("minDtPhiBQuality");
    edm::LogVerbatim("OMTFReconstruction")
        << "minDtPhiBQuality: " << edmParameterSet.getParameter<int>("minDtPhiBQuality") << std::endl;
  }
}
