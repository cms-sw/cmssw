/*
 * PatternOptimizer.h
 *
 *  Created on: Oct 12, 2017
 *      Author: kbunkow
 */

#ifndef L1T_OmtfP1_PATTERNOPTIMIZER_H_
#define L1T_OmtfP1_PATTERNOPTIMIZER_H_

#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/GoldenPatternWithStat.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/IOMTFEmulationObserver.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Tools/PatternOptimizerBase.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"

#include "TH1I.h"
#include "TH2I.h"

#include <functional>

class PatternOptimizer : public PatternOptimizerBase {
public:
  PatternOptimizer(const edm::ParameterSet& edmCfg,
                   const OMTFConfiguration* omtfConfig,
                   GoldenPatternVec<GoldenPatternWithStat>& gps);
  ~PatternOptimizer() override;

  /*  virtual void observeProcesorEmulation(unsigned int iProcessor, l1t::tftype mtfType,  const OMTFinput &input,
      const std::vector<AlgoMuon>& algoCandidates,
      std::vector<AlgoMuon>& gbCandidates,
      const std::vector<l1t::RegionalMuonCand> & candMuons);

  virtual void observeEventBegin(const edm::Event& iEvent);*/

  void observeEventEnd(const edm::Event& iEvent,
                       std::unique_ptr<l1t::RegionalMuonCandBxCollection>& finalCandidates) override;

  void endJob() override;

  const SimTrack* findSimMuon(const edm::Event& event, const SimTrack* previous = nullptr);

  static const unsigned int whatExptVal = 0;
  static const unsigned int whatExptNorm = 1;
  static const unsigned int whatOmtfVal = 2;
  static const unsigned int whatOmtfNorm = 3;

private:
  void saveHists(TFile& outfile) override;

  //candidate found by omtf in a given event
  //GoldenPatternResult omtfResult;
  GoldenPatternResult exptResult;

  double errorSum;
  int nEvents = 0;
  std::vector<double> errorSumRefL = std::vector<double>(8, 0.0);
  std::vector<double> nEventsRefL = std::vector<double>(8, 0.0);
  double maxdPdf = 0;
  std::string optPatXmlFile;

  unsigned int exptPatNum;

  unsigned int selectedPatNum;

  unsigned int currnetPtBatchPatNum;  //for threshold finding

  double deltaPdf = 0.01;

  //edm::Handle<edm::SimTrackContainer> simTks;

  std::vector<TH2I*> deltaPhi1_deltaPhi2_hits;
  std::vector<TH2I*> deltaPhi1_deltaPhi2_omtf;
  int selRefL, selL1, selL2;

  //std::vector<TH2I*> gpExpt_gpOmtf;
  //std::vector<TH1F*> gpEff;

  //double ptRangeFrom = 0;
  //double ptRangeTo = 0;

  unsigned int ptCut = 37;

  std::vector<double> rateWeights;

  std::vector<int> patternPtCodes;  //continous ptCode 1...31 (liek in the old PAC)

  std::vector<double> eventRateWeights;

  void initRateWeights();

  double getEventRateWeight(double pt) override;

  std::function<void(GoldenPatternWithStat* omtfCandGp, GoldenPatternWithStat* exptCandGp)> updateStatFunc;

  std::function<void(GoldenPatternWithStat* gp, unsigned int& iLayer, unsigned int& iRefLayer, double& learingRate)>
      updatePdfsFunc;

  void updateStatForAllGps(GoldenPatternWithStat* omtfCandGp, GoldenPatternWithStat* exptCandGp);

  void updateStat(GoldenPatternWithStat* omtfCandGp, GoldenPatternWithStat* exptCandGp, double delta, double norm);

  enum SecondCloserStatIndx { goodBigger, goodSmaller, badBigger, badSmaller };
  void updateStatCloseResults(GoldenPatternWithStat* omtfCandGp, GoldenPatternWithStat* exptCandGp);
  void updatePdfCloseResults();

  void updateStatCollectProb(GoldenPatternWithStat* omtfCandGp, GoldenPatternWithStat* exptCandGp);
  void calulateProb();

  void calculateThresholds(GoldenPatternWithStat* omtfCandGp, GoldenPatternWithStat* exptCandGp);
  void calculateThresholds(double targetEff);

  void tuneClassProb(GoldenPatternWithStat* omtfCandGp, GoldenPatternWithStat* exptCandGp);
  void tuneClassProb(double targetEff);

  //void updateStatPtDiff_1(GoldenPatternWithStat* omtfCandGp, GoldenPatternWithStat* exptCandGp);
  void updateStatVoter_1(GoldenPatternWithStat* omtfCandGp, GoldenPatternWithStat* exptCandGp);

  void updateStatPtDiff2_1(GoldenPatternWithStat* omtfCandGp, GoldenPatternWithStat* exptCandGp);

  void updateStatPtLogDiff_1(GoldenPatternWithStat* omtfCandGp, GoldenPatternWithStat* exptCandGp);
  void updateStatPtLogDiff_2(GoldenPatternWithStat* omtfCandGp, GoldenPatternWithStat* exptCandGp);

  void updatePdfsMean_1(GoldenPatternWithStat* gp, unsigned int& iLayer, unsigned int& iRefLayer, double& learingRate);
  void updatePdfsMean_2(GoldenPatternWithStat* gp, unsigned int& iLayer, unsigned int& iRefLayer, double& learingRate);
  void updatePdfsVoter_1(GoldenPatternWithStat* gp, unsigned int& iLayer, unsigned int& iRefLayer, double& learingRate);

  void updatePdfsAnaDeriv(GoldenPatternWithStat* gp, unsigned int& iLayer, unsigned int& iRefLayer, double& learingRate);
  void updatePdfsNumDeriv(GoldenPatternWithStat* gp, unsigned int& iLayer, unsigned int& iRefLayer, double& learingRate);

  void modifyPatterns();
  void modifyPatterns1(double step);
};

#endif /* L1T_OmtfP1_PATTERNOPTIMIZER_H_ */
