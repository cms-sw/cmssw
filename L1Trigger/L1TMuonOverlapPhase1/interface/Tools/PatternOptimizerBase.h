/*
 * PatternOptimizerBase.h
 *
 *  Created on: Oct 17, 2018
 *      Author: kbunkow
 */

#ifndef OMTF_PATTERNOPTIMIZERBASE_H_
#define OMTF_PATTERNOPTIMIZERBASE_H_

#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/GoldenPatternWithStat.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/IOMTFEmulationObserver.h"
#include <functional>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"

#include "TH1I.h"
#include "TH2I.h"
#include "TFile.h"

class PatternOptimizerBase : public IOMTFEmulationObserver {
public:
  static double vxMuRate(double pt_GeV);
  static double vxIntegMuRate(double pt_GeV, double dpt, double etaFrom, double etaTo);

  PatternOptimizerBase(const edm::ParameterSet& edmCfg, const OMTFConfiguration* omtfConfig);

  PatternOptimizerBase(const edm::ParameterSet& edmCfg,
                       const OMTFConfiguration* omtfConfig,
                       std::vector<std::shared_ptr<GoldenPatternWithStat> >& gps);

  ~PatternOptimizerBase() override;

  void observeProcesorEmulation(unsigned int iProcessor,
                                l1t::tftype mtfType,
                                const std::shared_ptr<OMTFinput>& input,
                                const AlgoMuons& algoCandidates,
                                const AlgoMuons& gbCandidates,
                                const std::vector<l1t::RegionalMuonCand>& candMuons) override;

  void observeEventBegin(const edm::Event& iEvent) override;

  void observeEventEnd(const edm::Event& iEvent,
                       std::unique_ptr<l1t::RegionalMuonCandBxCollection>& finalCandidates) override;

  void endJob() override;

  const SimTrack* findSimMuon(const edm::Event& event, const SimTrack* previous = nullptr);

protected:
  void savePatternsInRoot(std::string rootFileName);

  void printPatterns();

  virtual double getEventRateWeight(double pt) { return 1; }

  virtual void saveHists(TFile& outfile){};

  edm::ParameterSet edmCfg;
  const OMTFConfiguration* omtfConfig;

  std::vector<std::shared_ptr<GoldenPatternWithStat> > goldenPatterns;

  const SimTrack* simMuon = nullptr;

  //candidate found by omtf in a given event
  AlgoMuons::value_type omtfCand;
  l1t::RegionalMuonCand regionalMuonCand;

  AlgoMuons algoCandidates;

  unsigned int candProcIndx = 0;

  //GoldenPatternResult omtfResult;
  //GoldenPatternResult exptResult;

  TH1I* simMuPt;
  TH1I* simMuFoundByOmtfPt;

  TH1F* simMuPtSpectrum;

  bool writeLayerStat = false;
};

#endif /* OMTF_PATTERNOPTIMIZERBASE_H_ */
