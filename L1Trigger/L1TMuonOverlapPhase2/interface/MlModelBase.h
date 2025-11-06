/*
 * PtAssignmentBase.h
 *
 *  Created on: Mar 16, 2020
 *      Author: kbunkow
 */

#ifndef L1T_OmtfP2_MlModelBase_H_
#define L1T_OmtfP2_MlModelBase_H_

#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/AlgoMuon.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/IOMTFEmulationObserver.h"

/*
 * base class for the objects providing an alternative pt assignment on top of the OMTF golden pattern (like neural network)
 */
class MlModelBase {
public:
  MlModelBase(const OMTFConfiguration* omtfConfig) : omtfConfig(omtfConfig) {}
  virtual ~MlModelBase();

  virtual void run(AlgoMuons::value_type& algoMuon,
                   std::vector<std::unique_ptr<IOMTFEmulationObserver> >& observers) = 0;

protected:
  const OMTFConfiguration* omtfConfig = nullptr;
};

#endif /* L1T_OmtfP2_MlModelBase_H_ */
