/*
 * ProcConfigurationBase.h
 *
 *  Created on: Jan 30, 2019
 *      Author: Karol Bunkowski kbunkow@cern.ch
 */

#ifndef L1T_OmtfP1_PROCCONFIGURATIONBASE_H_
#define L1T_OmtfP1_PROCCONFIGURATIONBASE_H_
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <cmath>

class ProcConfigurationBase {
public:
  ProcConfigurationBase();
  virtual ~ProcConfigurationBase();

  /**configuration from the edm::ParameterSet
   * the parameters are set (i.e. overwritten) only if their exist in the edmParameterSet
   */
  virtual void configureFromEdmParameterSet(const edm::ParameterSet& edmParameterSet);

  virtual unsigned int nPhiBins() const = 0;

  virtual unsigned int nProcessors() const = 0;

  virtual double hwPtToGev(int hwPt) const = 0;

  ///uGMT pt scale conversion: [0GeV, 0.5GeV) = 1 [0.5GeV, 1 Gev) = 2
  virtual int ptGevToHw(double ptGev) const = 0;

  //processor internal phi scale
  virtual int getProcScalePhi(double phiRad, double procPhiZeroRad = 0) const = 0;

  virtual int etaToHwEta(double eta) const = 0;

  //eta of the middle DT Wheel 2 MB1, in the OMTF scale
  virtual int mb1W2Eta() const = 0;

  //eta of the middle DT Wheel 2 MB2, in the OMTF scale
  virtual int mb2W2Eta() const = 0;

  //eta of the middle DT Wheel 2 MB3, in the OMTF scale
  virtual int mb3W2Eta() const = 0;

  //eta of the middle DT Wheel 2 MB4, in the OMTF scale
  virtual int mb4W2Eta() const = 0;

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

  virtual int dtBxShift() const { return dtBxShift_; }

  virtual void setDtBxShift(int dtBxShift) { this->dtBxShift_ = dtBxShift; }

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

  enum class StubEtaEncoding {
    //in the firmware the eta is encoded as fired bits in the 9bit word, this is DT phase-1 encoding.
    //In the emulator in most of the places eta value is used, but with the DT-like bining, i.e. only certain values are valid, see OMTFConfiguration::eta2Bits()
    //this is the OMTF run2 option
    bits = 0,
    //the phase1 eta scale is used, but all hw values are valid, i.e. the DT-phase-1 bining is NOT used
    valueP1Scale = 1,
    valueP2Scale = 2,
  };

  StubEtaEncoding getStubEtaEncoding() const { return stubEtaEncoding; }

  void setStubEtaEncoding(StubEtaEncoding stubEtaEncoding) { this->stubEtaEncoding = stubEtaEncoding; }

  //[unit/rad] for DT segment phiB, as it is at the level of the algorithm
  //in the link data it can be different, and it is converted in the DtDigiToStubsConverterOmtf::addDTphiDigi
  double dtPhiBUnitsRad() const { return dtPhiBUnitsRad_; }

  double etaUnit() const { return etaUnit_; }

protected:
  double etaUnit_ = 0.010875;  //=2.61/240 - value from the phase1 interface note

private:
  int cscLctCentralBx_ = 8;  //CSCConstants::LCT_CENTRAL_BX;

  int dtBxShift_ = 20;  //phase-2 DT segment BX shift, different for MC and data

  //parameters of the RpcClusterization
  unsigned int rpcMaxClusterSize = 3;
  unsigned int rpcMaxClusterCnt = 2;

  bool rpcDropAllClustersIfMoreThanMax = false;
  // if true no  cluster is return if there is more clusters then maxClusterCnt (counted regardless of the size)

  int minDtPhiQuality = 2;

  int minDtPhiBQuality = 2;  //used on the top of the minDtPhiQuality

  double dtPhiBUnitsRad_ = 512;  //[unit/rad] for DT segment phiB, it is at the level of the algorithm, not inputs

  bool fixCscGeometryOffset = false;

  StubEtaEncoding stubEtaEncoding = StubEtaEncoding::bits;
};

#endif /* L1T_OmtfP1_PROCCONFIGURATIONBASE_H_ */
