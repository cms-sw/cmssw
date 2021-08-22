/*
 * ProcConfigurationBase.h
 *
 *  Created on: Jan 30, 2019
 *      Author: Karol Bunkowski kbunkow@cern.ch
 */

#ifndef L1T_OmtfP1_PROCCONFIGURATIONBASE_H_
#define L1T_OmtfP1_PROCCONFIGURATIONBASE_H_
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class ProcConfigurationBase {
public:
  ProcConfigurationBase();
  virtual ~ProcConfigurationBase();

  /**configuration from the edm::ParameterSet
   * the parameters are set (i.e. overwritten) only if their exist in the edmParameterSet
   */
  virtual void configureFromEdmParameterSet(const edm::ParameterSet& edmParameterSet);

  virtual unsigned int nPhiBins() const = 0;

  virtual double hwPtToGev(int hwPt) const = 0;

  ///uGMT pt scale conversion: [0GeV, 0.5GeV) = 1 [0.5GeV, 1 Gev) = 2
  virtual int ptGevToHw(double ptGev) const = 0;

  //processor internal phi scale
  virtual int getProcScalePhi(double phiRad, double procPhiZeroRad = 0) const = 0;

  virtual int etaToHwEta(double eta) const = 0;

  //returns address for the  pdf LUTs
  virtual unsigned int ptHwToPtBin(int ptHw) const { return 0; }

  virtual unsigned int ptGeVToPtBin(float ptGeV) const { return 0; }

  //returns address for the  pdf LUTs
  virtual unsigned int etaHwToEtaBin(int etaHw) const { return 0; }

  virtual int foldPhi(int phi) const;

  virtual unsigned int nLayers() const = 0;

  virtual bool isBendingLayer(unsigned int iLayer) const = 0;

  virtual unsigned int getBxToProcess() const { return 1; }

  virtual int cscLctCentralBx() const { return cscLctCentralBx_; }

  virtual void setCscLctCentralBx(int lctCentralBx) { this->cscLctCentralBx_ = lctCentralBx; }

  virtual bool getRpcDropAllClustersIfMoreThanMax() const { return rpcDropAllClustersIfMoreThanMax; }

  virtual void setRpcDropAllClustersIfMoreThanMax(bool rpcDropAllClustersIfMoreThanMax = true) {
    this->rpcDropAllClustersIfMoreThanMax = rpcDropAllClustersIfMoreThanMax;
  }

  virtual unsigned int getRpcMaxClusterCnt() const { return rpcMaxClusterCnt; }

  virtual void setRpcMaxClusterCnt(unsigned int rpcMaxClusterCnt = 2) { this->rpcMaxClusterCnt = rpcMaxClusterCnt; }

  virtual unsigned int getRpcMaxClusterSize() const { return rpcMaxClusterSize; }

  virtual void setRpcMaxClusterSize(unsigned int rpcMaxClusterSize = 4) { this->rpcMaxClusterSize = rpcMaxClusterSize; }

  virtual int getMinDtPhiQuality() const { return minDtPhiQuality; }

  virtual void setMinDtPhiQuality(int minDtPhiQuality = 2) { this->minDtPhiQuality = minDtPhiQuality; }

  virtual int getMinDtPhiBQuality() const { return minDtPhiBQuality; }

  virtual void setMinDtPhiBQuality(int minDtPhiBQuality = 2) { this->minDtPhiBQuality = minDtPhiBQuality; }

  virtual bool getFixCscGeometryOffset() const { return fixCscGeometryOffset; }

  virtual bool setFixCscGeometryOffset(bool fixCscGeometryOffset) {
    return this->fixCscGeometryOffset = fixCscGeometryOffset;
  }

private:
  int cscLctCentralBx_ = 8;  //CSCConstants::LCT_CENTRAL_BX;

  //parameters of the RpcClusterization
  unsigned int rpcMaxClusterSize = 3;
  unsigned int rpcMaxClusterCnt = 2;

  bool rpcDropAllClustersIfMoreThanMax = false;
  // if true no  cluster is return if there is more clusters then maxClusterCnt (counted regardless of the size)

  int minDtPhiQuality = 2;

  int minDtPhiBQuality = 2;  //used on the top of the minDtPhiQuality

  bool fixCscGeometryOffset = false;
};

#endif /* L1T_OmtfP1_PROCCONFIGURATIONBASE_H_ */
