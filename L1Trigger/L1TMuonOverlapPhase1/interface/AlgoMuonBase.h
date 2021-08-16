/*
 * AlgoMuonBase.h
 *
 *  Created on: Mar 1, 2019
 *      Author: Karol Bunkowski kbunkow@cern.ch
 */

#ifndef INTERFACE_ALGOMUONBASE_H_
#define INTERFACE_ALGOMUONBASE_H_

#include "L1Trigger/L1TMuonOverlapPhase1/interface/ProcConfigurationBase.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/MuonStub.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/StubResult.h"
#include "boost/dynamic_bitset.hpp"

class AlgoMuonBase {
public:
  AlgoMuonBase(){};

  AlgoMuonBase(const ProcConfigurationBase* config);
  virtual ~AlgoMuonBase();

  virtual int getEtaHw() const = 0;

  virtual bool isValid() const = 0;

  //virtual void setValid(bool valid) = 0;

  virtual unsigned int getFiredLayerCnt() const {
    unsigned int count = 0;
    for (auto& firedLayerBits : firedLayerBitsInBx) {
      count += firedLayerBits.count();
    }
    return count;
  }

  virtual unsigned int getFiredLayerCnt(int bx) const { return firedLayerBitsInBx.at(bx).count(); }

  boost::dynamic_bitset<> getFiredLayerBits()
      const {  //TODO make it virtual, and change the return type in in the AlgoMuon to dynamic_bitset<>
    boost::dynamic_bitset<> firedLayerBitsSum(firedLayerBitsInBx[0].size());
    for (auto& firedLayerBits : firedLayerBitsInBx) {
      firedLayerBitsSum |= firedLayerBits;
    }
    return firedLayerBitsSum;
  }

  virtual bool isLayerFired(unsigned int iLayer, unsigned int bx) const { return firedLayerBitsInBx.at(bx)[iLayer]; }

  virtual double getPdfSum() const = 0;

  /* virtual const bool isKilled() const = 0;

  virtual void kill()  = 0;*/

  //virtual bool isLayerFired(unsigned int iLayer) const  = 0;

  virtual const StubResult& getStubResult(unsigned int iLayer) const = 0;

  virtual const StubResults& getStubResults() const = 0;

  //virtual const boost::dynamic_bitset<>& getFiredLayerBits() const  = 0;

  virtual void setBeta(float beta) { this->beta = beta; }

  virtual float getBeta() const { return beta; }

  virtual double getSimBeta() const { return 0; }

  virtual float getBetaLikelihood() const { return betaLikelihood; }

  virtual void setBetaLikelihood(float betaLikelihood = 0) { this->betaLikelihood = betaLikelihood; }

protected:
  float beta = 0;            //zero means it is not measured
  float betaLikelihood = 0;  //beta measurement goodness, likelihood of return beta hypothesis

  ///bits representing fired logicLayers (including bending layers),
  std::vector<boost::dynamic_bitset<> > firedLayerBitsInBx;
};

#endif /* INTERFACE_ALGOMUONBASE_H_ */
