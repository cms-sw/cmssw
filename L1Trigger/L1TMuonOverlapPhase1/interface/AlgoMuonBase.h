/*
 * AlgoMuonBase.h
 *
 *  Created on: Mar 1, 2019
 *      Author: Karol Bunkowski kbunkow@cern.ch
 */

#ifndef L1T_OmtfP1_ALGOMUONBASE_H_
#define L1T_OmtfP1_ALGOMUONBASE_H_

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

  virtual const StubResult& getStubResult(unsigned int iLayer) const = 0;

  virtual const StubResults& getStubResults() const = 0;

protected:
  ///bits representing fired logicLayers (including bending layers),
  std::vector<boost::dynamic_bitset<> > firedLayerBitsInBx;
};

#endif /* L1T_OmtfP1_ALGOMUONBASE_H_ */
