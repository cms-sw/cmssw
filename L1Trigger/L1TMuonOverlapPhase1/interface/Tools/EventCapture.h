/*
 * EventCapture.h
 *
 *  Created on: Oct 23, 2019
 *      Author: kbunkow
 */

#ifndef L1T_OmtfP1_EVENTCAPTURE_H_
#define L1T_OmtfP1_EVENTCAPTURE_H_

#include "SimDataFormats/Track/interface/SimTrack.h"

#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/IOMTFEmulationObserver.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/GoldenPatternWithStat.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Tools/CandidateSimMuonMatcher.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Tools/StubsSimHitsMatcher.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class EventCapture : public IOMTFEmulationObserver {
public:
  EventCapture(const edm::ParameterSet& edmCfg,
               const OMTFConfiguration* omtfConfig,
               CandidateSimMuonMatcher* candidateSimMuonMatcher,
               const MuonGeometryTokens& muonGeometryTokens,
               const GoldenPatternVec<GoldenPattern>* gps = nullptr);

  ~EventCapture() override;

  void beginRun(edm::EventSetup const& eventSetup) override;

  void observeProcesorEmulation(unsigned int iProcessor,
                                l1t::tftype mtfType,
                                const std::shared_ptr<OMTFinput>&,
                                const AlgoMuons& algoCandidates,
                                const AlgoMuons& gbCandidates,
                                const std::vector<l1t::RegionalMuonCand>& candMuons) override;

  void observeEventBegin(const edm::Event& event) override;

  void observeEventEnd(const edm::Event& event,
                       std::unique_ptr<l1t::RegionalMuonCandBxCollection>& finalCandidates) override;

  void endJob() override;

private:
  edm::InputTag simTracksTag;
  const OMTFConfiguration* omtfConfig = nullptr;

  const GoldenPatternVec<GoldenPattern>* goldenPatterns = nullptr;

  CandidateSimMuonMatcher* candidateSimMuonMatcher = nullptr;

  std::vector<edm::Ptr<SimTrack> > simMuons;

  std::vector<std::shared_ptr<OMTFinput> > inputInProcs;
  std::vector<AlgoMuons> algoMuonsInProcs;
  std::vector<AlgoMuons> gbCandidatesInProcs;

  std::unique_ptr<StubsSimHitsMatcher> stubsSimHitsMatcher;
};

#endif /* L1T_OmtfP1_EVENTCAPTURE_H_ */
