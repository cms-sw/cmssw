/*
 * PatternOptimizerBase.h
 *
 *  Created on: Oct 17, 2018
 *      Author: kbunkow
 */

#ifndef OMTF_PATTERNOPTIMIZERBASE_H_
#define OMTF_PATTERNOPTIMIZERBASE_H_

#include <L1Trigger/L1TMuonBayes/interface/Omtf/GoldenPatternWithStat.h>
#include <L1Trigger/L1TMuonBayes/interface/Omtf/IOMTFEmulationObserver.h>
#include <functional>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"

#include "TH1I.h"
#include "TH2I.h"

class PatternOptimizerBase: public IOMTFEmulationObserver {
public:
  PatternOptimizerBase(const edm::ParameterSet& edmCfg, const OMTFConfiguration* omtfConfig, std::vector<std::shared_ptr<GoldenPatternWithStat> >& gps);

  virtual ~PatternOptimizerBase();

  virtual void observeProcesorEmulation(unsigned int iProcessor, l1t::tftype mtfType,  const OMTFinput &input,
      const AlgoMuons& algoCandidates,
      const AlgoMuons& gbCandidates,
      const std::vector<l1t::RegionalMuonCand> & candMuons);

  virtual void observeEventBegin(const edm::Event& iEvent);

  virtual void observeEventEnd(const edm::Event& iEvent);

  virtual void endJob();

  const SimTrack* findSimMuon(const edm::Event &event, const SimTrack* previous = 0);

protected:
  void savePatternsInRoot(std::string rootFileName);

  void printPatterns();

  virtual double getEventRateWeight(double pt) {
    return 1;
  }

  virtual void saveHists(TFile& outfile) {};

  edm::ParameterSet edmCfg;
  const OMTFConfiguration* omtfConfig;

  std::vector<std::shared_ptr<GoldenPatternWithStat> > goldenPatterns;

  const SimTrack* simMuon = nullptr;

  //candidate found by omtf in a given event
  AlgoMuons::value_type omtfCand;
  l1t::RegionalMuonCand regionalMuonCand;

  unsigned int candProcIndx = 0;

  //GoldenPatternResult omtfResult;
  //GoldenPatternResult exptResult;

  TH1I* simMuPt;
  TH1I* simMuFoundByOmtfPt;

  TH1F* simMuPtSpectrum;
};

#endif /* OMTF_PATTERNOPTIMIZERBASE_H_ */
