/*
 * PatternsPtAssignment.h
 *
 *  Created on: Mar 9, 2020
 *      Author: kbunkow
 */

#ifndef INTERFACE_TOOLS_PATTERNSPTASSIGNMENT_H_
#define INTERFACE_TOOLS_PATTERNSPTASSIGNMENT_H_

#include "L1Trigger/L1TMuonOverlapPhase1/interface/Tools/GpResultsToPt.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Tools/PatternOptimizerBase.h"

class PatternsPtAssignment : public PatternOptimizerBase {
public:
  PatternsPtAssignment(const edm::ParameterSet& edmCfg,
                       const OMTFConfiguration* omtfConfig,
                       const std::vector<std::shared_ptr<GoldenPattern> >& gps,
                       std::string rootFileName);

  ~PatternsPtAssignment() override;

  void observeEventEnd(const edm::Event& iEvent,
                       std::unique_ptr<l1t::RegionalMuonCandBxCollection>& finalCandidates) override;

  void endJob() override;

private:
  std::vector<std::shared_ptr<GoldenPattern> > gps;

  GpResultsToPt* gpResultsToPt = nullptr;
};

#endif /* INTERFACE_TOOLS_PATTERNSPTASSIGNMENT_H_ */
