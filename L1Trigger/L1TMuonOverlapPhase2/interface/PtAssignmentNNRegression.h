/*
 * PtAssignmentNN.h
 *
 *  Created on: May 8, 2020
 *      Author: kbunkow
 */

#ifndef INTERFACE_OMTF_PTASSIGNMENTNN_H_
#define INTERFACE_OMTF_PTASSIGNMENTNN_H_

#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/PtAssignmentBase.h"
#include "L1Trigger/L1TMuonOverlapPhase2/interface/LutNetworkFixedPointRegression2Outputs.h"

//#include "lutNN/lutNN2/interface/ClassifierToRegression.h"

class PtAssignmentNNRegression : public PtAssignmentBase {
public:
  PtAssignmentNNRegression(const edm::ParameterSet& edmCfg,
                           const OMTFConfiguration* omtfConfig,
                           std::string networkFile);
  ~PtAssignmentNNRegression() override;

  std::vector<float> getPts(AlgoMuons::value_type& algoMuon,
                            std::vector<std::unique_ptr<IOMTFEmulationObserver> >& observers) override;

private:
  unique_ptr<lutNN::LutNetworkFixedPointRegressionBase> lutNetworkFP;
};

#endif /* INTERFACE_OMTF_PTASSIGNMENTNN_H_ */
