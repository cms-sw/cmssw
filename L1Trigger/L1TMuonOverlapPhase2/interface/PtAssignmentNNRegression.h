/*
 * PtAssignmentNN.h
 *
 *  Created on: May 8, 2020
 *      Author: kbunkow
 */

#ifndef L1Trigger_L1TMuonOverlapPhase2_PtAssigmentNNRegression_h
#define L1Trigger_L1TMuonOverlapPhase2_PtAssigmentNNRegression_h

#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/PtAssignmentBase.h"
#include "L1Trigger/L1TMuonOverlapPhase2/interface/LutNetworkFixedPointRegression2Outputs.h"

class PtAssignmentNNRegression : public PtAssignmentBase {
public:
  PtAssignmentNNRegression(const edm::ParameterSet& edmCfg,
                           const OMTFConfiguration* omtfConfig,
                           std::string networkFile);
  ~PtAssignmentNNRegression() override = default;

  std::vector<float> getPts(AlgoMuons::value_type& algoMuon,
                            std::vector<std::unique_ptr<IOMTFEmulationObserver> >& observers) override;

private:
  unique_ptr<lutNN::LutNetworkFixedPointRegressionBase> lutNetworkFP;
};

#endif /* L1Trigger_L1TMuonOverlapPhase2_PtAssigmentNNRegression_h */
