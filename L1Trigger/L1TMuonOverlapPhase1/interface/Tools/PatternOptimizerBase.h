/*
 * PatternOptimizerBase.h
 *
 *  Created on: Oct 17, 2018
 *      Author: kbunkow
 */

#ifndef L1T_OmtfP1_PATTERNOPTIMIZERBASE_H_
#define L1T_OmtfP1_PATTERNOPTIMIZERBASE_H_

#include "L1Trigger/L1TMuonOverlapPhase1/interface/Tools/EmulationObserverBase.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/GoldenPatternWithStat.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"

#include "TH1I.h"
#include "TH2I.h"
#include "TFile.h"

#include <functional>

class PatternOptimizerBase : public EmulationObserverBase {
public:
  static double vxMuRate(double pt_GeV);
  static double vxIntegMuRate(double pt_GeV, double dpt, double etaFrom, double etaTo);

  PatternOptimizerBase(const edm::ParameterSet& edmCfg,
                       const OMTFConfiguration* omtfConfig,
                       GoldenPatternVec<GoldenPatternWithStat>& gps);

  ~PatternOptimizerBase() override;

  /*  void observeProcesorEmulation(unsigned int iProcessor,
                                l1t::tftype mtfType,
                                const std::shared_ptr<OMTFinput>& input,
                                const AlgoMuons& algoCandidates,
                                const AlgoMuons& gbCandidates,
                                const std::vector<l1t::RegionalMuonCand>& candMuons) override;

  void observeEventBegin(const edm::Event& iEvent) override;*/

  void observeEventEnd(const edm::Event& iEvent,
                       std::unique_ptr<l1t::RegionalMuonCandBxCollection>& finalCandidates) override;

  void endJob() override;

protected:
  void savePatternsInRoot(std::string rootFileName);

  void printPatterns();

  virtual double getEventRateWeight(double pt) { return 1; }

  virtual void saveHists(TFile& outfile){};

  GoldenPatternVec<GoldenPatternWithStat>& goldenPatterns;

  //GoldenPatternResult omtfResult;
  //GoldenPatternResult exptResult;

  TH1I* simMuPt;
  TH1I* simMuFoundByOmtfPt;

  TH1F* simMuPtSpectrum;

  bool writeLayerStat = false;
};

#endif /* L1T_OmtfP1_PATTERNOPTIMIZERBASE_H_ */
