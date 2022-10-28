/*
 * MuonStubsInput.h
 *
 *  Created on: Jan 31, 2019
 *      Author: Karol Bunkowski kbunkow@cern.ch
 */

#ifndef L1T_OmtfP1_MUONSTUBSINPUT_H_
#define L1T_OmtfP1_MUONSTUBSINPUT_H_

#include "L1Trigger/L1TMuonOverlapPhase1/interface/ProcConfigurationBase.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/MuonStub.h"
#include <ostream>

class MuonStubsInput {
public:
  MuonStubsInput(const ProcConfigurationBase* config);

  virtual ~MuonStubsInput() {}

  virtual void addStub(unsigned int iLayer, const MuonStubPtr& stub) {
    muonStubsInLayers.at(iLayer).emplace_back(stub);
  }

  virtual MuonStubPtrs2D& getMuonStubs() { return muonStubsInLayers; }

  virtual const MuonStubPtrs2D& getMuonStubs() const { return muonStubsInLayers; }

  //gives stub phiHw or phiBHw - depending which layer is requested
  //if there is no stub at input iInput - return MuonStub::EMTPY_PHI
  virtual int getPhiHw(unsigned int iLayer, unsigned int iInput) const;

  friend std::ostream& operator<<(std::ostream& out, const MuonStubsInput& stubsInput);

protected:
  const ProcConfigurationBase* config = nullptr;

  //indexing: muonStubsInLayers[iLayer][iStub]
  MuonStubPtrs2D muonStubsInLayers;
};

#endif /* L1T_OmtfP1_MUONSTUBSINPUT_H_ */
