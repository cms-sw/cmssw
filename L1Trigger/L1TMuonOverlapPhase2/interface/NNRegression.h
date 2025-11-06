/*
 * PtAssignmentNN.h
 *
 *  Created on: May 8, 2020
 *      Author: kbunkow
 */

#ifndef L1Trigger_L1TMuonOverlapPhase2_PtAssigmentNNRegression_h
#define L1Trigger_L1TMuonOverlapPhase2_PtAssigmentNNRegression_h

#include "L1Trigger/L1TMuonOverlapPhase2/interface/MlModelBase.h"
#include "L1Trigger/L1TMuonOverlapPhase2/interface/LutNetworkFixedPointCommon.h"

class NNRegression : public MlModelBase {
public:
  NNRegression(const edm::ParameterSet& edmCfg,
                           const OMTFConfiguration* omtfConfig,
                           std::string networkFile);
  ~NNRegression() override = default;

  void run(AlgoMuons::value_type& algoMuon, std::vector<std::unique_ptr<IOMTFEmulationObserver> >& observers) override;

private:
  unique_ptr<lutNN::LutNetworkFixedPointRegressionBase> lutNetworkFP;
};

#endif /* L1Trigger_L1TMuonOverlapPhase2_PtAssigmentNNRegression_h */
