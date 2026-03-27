/*
 * MuonCandidateMatcher.h
 *
 *  Created on: Dec 14, 2020
 *      Author: kbunkow
 */

#ifndef L1T_OmtfP1_TOOLS_MUONCANDIDATEMATCHER_H_
#define L1T_OmtfP1_TOOLS_MUONCANDIDATEMATCHER_H_

#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/AlgoMuon.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/IOMTFEmulationObserver.h"

////////////////////
// FRAMEWORK HEADERS
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

///////////////////////
// DATA FORMATS HEADERS
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"

#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

////////////////////////////
// DETECTOR GEOMETRY HEADERS
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

////////////////
// PHYSICS TOOLS
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixStateInfo.h"

#include "TH1D.h"

class MatchingResult {
public:
  enum class ResultType : short { propagationFailed = -1, notMatched = 0, matched = 1, duplicate = 2 };

  MatchingResult() {}

  MatchingResult(const SimTrack& simTrack, const SimVertex* simVertex);

  MatchingResult(const TrackingParticle& trackingParticle);

  ResultType result = ResultType::notMatched;
  //bool propagationFailed = false;
  double deltaPhi = 0;
  double deltaEta = 0;

  double propagatedPhi = 0;
  double propagatedEta = 0;

  double matchingLikelihood = 0;

  FinalMuonPtr muonCand;

  //to avoid using simTrack or trackingParticle
  int pdgId = 0;
  //int parrentPdgId = 0;
  double genPt = 0;
  double genEta = 0;
  double genPhi = 0;
  int genCharge = 0;

  float vertexEta = 0;
  float vertexPhi = 0;
  float muonDxy = 0;
  float muonRho = 0;
  int parentPdgId = 0;

  const SimTrack* simTrack = nullptr;
  const SimVertex* simVertex = nullptr;

  const TrackingParticle* trackingParticle = nullptr;

  friend std::ostream& operator<<(std::ostream& out, const MatchingResult& matchingResult);
};

/*
 * matches simMuons or tracking particles
 */
class CandidateSimMuonMatcher : public IOMTFEmulationObserver {
public:
  CandidateSimMuonMatcher(const edm::ParameterSet& edmCfg,
                          //const OMTFConfiguration* omtfConfig,
                          int nProcessors,
                          const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord>& magneticFieldEsToken,
                          const edm::ESGetToken<Propagator, TrackingComponentsRecord>& propagatorEsToken);

  ~CandidateSimMuonMatcher() override;

  void beginRun(edm::EventSetup const& eventSetup) override;

  void observeProcesorEmulation(unsigned int iProcessor,
                                l1t::tftype mtfType,
                                const std::shared_ptr<OMTFinput>&,
                                const AlgoMuons& algoCandidates,
                                const AlgoMuons& gbCandidates,
                                const FinalMuons& finalMuons) override;

  void observeEventBegin(const edm::Event& event) override;

  void observeEventEnd(const edm::Event& event, FinalMuons& finalMuons) override;

  void endJob() override;

  int calcGlobalPhi(int locPhi, int proc);

  //simplified ghost busting
  FinalMuons ghostBust(const FinalMuons& finalMuons);

  FreeTrajectoryState simTrackToFts(const SimTrack& simTrack, const SimVertex& simVertex);

  FreeTrajectoryState simTrackToFts(const TrackingParticle& trackingParticle);

  TrajectoryStateOnSurface atStation1(const FreeTrajectoryState& ftsStart) const;

  TrajectoryStateOnSurface atStation2(const FreeTrajectoryState& ftsStart) const;

  TrajectoryStateOnSurface propagate(const SimTrack& simTrack, const edm::SimVertexContainer* simVertices);

  TrajectoryStateOnSurface propagate(const TrackingParticle& trackingParticle);

  void propagate(MatchingResult& result);

  void match(const FinalMuons& finalMuons, MatchingResult& result, std::vector<MatchingResult>& matchingResults);

  void match(const FinalMuonPtr& finalMuon3, MatchingResult& result);

  std::vector<MatchingResult> cleanMatching(std::vector<MatchingResult> matchingResults, const FinalMuons& finalMuons);

  std::vector<MatchingResult> match(const FinalMuons& finalMuons,
                                    const edm::SimTrackContainer* simTracks,
                                    const edm::SimVertexContainer* simVertices,
                                    std::function<bool(const SimTrack&)> const& simTrackFilter);

  std::vector<MatchingResult> match(const FinalMuons& finalMuons,
                                    const TrackingParticleCollection* trackingParticles,
                                    std::function<bool(const TrackingParticle&)> const& simTrackFilter);

  //matching without any propagation, just checking basic geometrical agreement between simMuon and candidates
  //problem with propagation is the it does not work for low pt muons (pt < ~3GeV)
  //which is not good for dumping the data for the NN training. So for that purpose it is better to use the matchSimple
  std::vector<MatchingResult> matchSimple(const FinalMuons& finalMuons,
                                          const edm::SimTrackContainer* simTracks,
                                          const edm::SimVertexContainer* simVertices,
                                          std::function<bool(const SimTrack&)> const& simTrackFilter);

  //no matching, just collect muonCands
  std::vector<MatchingResult> collectMuonCands(const FinalMuons& finalMuons);

  std::vector<MatchingResult> getMatchingResults() { return matchingResults; }

  enum class MatchingType : short {
    noMatcher = -1,
    withPropagator = 0,
    simplePropagation = 1,
    simpleMatching = 2,
    collectMuonCands = 3
  };

  MatchingType getMatchingType() const { return matchingType; }

private:
  int nProcessors = 6;

  const edm::ParameterSet& edmCfg;

  AlgoMuons gbCandidates;
  std::vector<MatchingResult> matchingResults;

  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord>& magneticFieldEsToken;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord>& propagatorEsToken;

  edm::ESHandle<MagneticField> magField;
  edm::ESHandle<Propagator> propagator;

  TH1* minDelta_pos = nullptr;
  TH1* maxDelta_pos = nullptr;
  TH1* medianDelta_pos = nullptr;

  TH1* minDelta_neg = nullptr;
  TH1* maxDelta_neg = nullptr;
  TH1* medianDelta_neg = nullptr;

  MatchingType matchingType = MatchingType::simpleMatching;
  //bool usePropagation = false;
};

#endif /* L1T_OmtfP1_TOOLS_MUONCANDIDATEMATCHER_H_ */
