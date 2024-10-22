/*
 * PatternGenerator.h
 *
 *  Created on: Nov 8, 2019
 *      Author: kbunkow
 */

#ifndef L1T_OmtfP1_PATTERNGENERATOR_H_
#define L1T_OmtfP1_PATTERNGENERATOR_H_

#include "L1Trigger/L1TMuonOverlapPhase1/interface/Tools/PatternOptimizerBase.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Tools/CandidateSimMuonMatcher.h"

class PatternGenerator : public PatternOptimizerBase {
public:
  PatternGenerator(const edm::ParameterSet& edmCfg,
                   const OMTFConfiguration* omtfConfig,
                   GoldenPatternVec<GoldenPatternWithStat>& gps,
                   CandidateSimMuonMatcher* candidateSimMuonMatcher);

  ~PatternGenerator() override;

  void observeEventEnd(const edm::Event& iEvent,
                       std::unique_ptr<l1t::RegionalMuonCandBxCollection>& finalCandidates) override;

  void endJob() override;

protected:
  void initPatternGen();

  void updateStat();

  void updateStatUsingMatcher2();

  std::function<void()> updateStatFunction;

  void upadatePdfs();

  void saveHists(TFile& outfile) override;

  void modifyClassProb(double step);

  void reCalibratePt();

  void groupPatterns();

  CandidateSimMuonMatcher* candidateSimMuonMatcher = nullptr;

  //indexing: [charge][iLayer]
  std::vector<std::vector<TH2I*> > ptDeltaPhiHists;

  std::vector<unsigned int> eventCntPerGp;
};

#endif /* L1T_OmtfP1_PATTERNGENERATOR_H_ */
