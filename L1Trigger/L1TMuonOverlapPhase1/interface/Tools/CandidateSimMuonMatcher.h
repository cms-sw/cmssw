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

  MatchingResult(const SimTrack& simTrack) : simTrack(&simTrack) {
    pdgId = simTrack.type();
    genPt = simTrack.momentum().pt();
    genEta = simTrack.momentum().eta();
    genPhi = simTrack.momentum().phi();
  }

  MatchingResult(const TrackingParticle& trackingParticle) : trackingParticle(&trackingParticle) {
    pdgId = trackingParticle.pdgId();
    genPt = trackingParticle.pt();
    genEta = trackingParticle.momentum().eta();
    genPhi = trackingParticle.momentum().phi();
  }

  ResultType result = ResultType::notMatched;
  //bool propagationFailed = false;
  double deltaPhi = 0;
  double deltaEta = 0;

  double propagatedPhi = 0;
  double propagatedEta = 0;

  double matchingLikelihood = 0;

  const l1t::RegionalMuonCand* muonCand = nullptr;
  AlgoMuonPtr procMuon;  //Processor gbCandidate

  //to avoid using simTrack or trackingParticle
  double pdgId = 0;
  double genPt = 0;
  double genEta = 0;
  double genPhi = 0;

  const SimTrack* simTrack = nullptr;
  const SimVertex* simVertex = nullptr;

  const TrackingParticle* trackingParticle = nullptr;
};

/*
 * matches simMuons or tracking particles
 */
class CandidateSimMuonMatcher : public IOMTFEmulationObserver {
public:
  CandidateSimMuonMatcher(const edm::ParameterSet& edmCfg,
                          const OMTFConfiguration* omtfConfig,
                          const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord>& magneticFieldEsToken,
                          const edm::ESGetToken<Propagator, TrackingComponentsRecord>& propagatorEsToken);

  ~CandidateSimMuonMatcher() override;

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

  //simplified ghost busting
  //only candidates in the bx=0 are included
  //ghost busts at the same time the  mtfCands and the gbCandidates
  //gbCandidates - all gbCandidates from all processors, should be one-to-one as the mtfCands,
  //and the ghostBustedProcMuons are one-to-onr to the returned RegionalMuonCands
  std::vector<const l1t::RegionalMuonCand*> ghostBust(const l1t::RegionalMuonCandBxCollection* mtfCands,
                                                      const AlgoMuons& gbCandidates,
                                                      AlgoMuons& ghostBustedProcMuons);

  FreeTrajectoryState simTrackToFts(const SimTrack& simTrack, const SimVertex& simVertex);

  FreeTrajectoryState simTrackToFts(const TrackingParticle& trackingParticle);

  TrajectoryStateOnSurface atStation2(const FreeTrajectoryState& ftsStart) const;

  TrajectoryStateOnSurface propagate(const SimTrack& simTrack, const edm::SimVertexContainer* simVertices);

  TrajectoryStateOnSurface propagate(const TrackingParticle& trackingParticle);

  //tsof should be the result of track propagation
  MatchingResult match(const l1t::RegionalMuonCand* omtfCand,
                       const AlgoMuonPtr& procMuon,
                       const SimTrack& simTrack,
                       TrajectoryStateOnSurface& tsof);

  MatchingResult match(const l1t::RegionalMuonCand* omtfCand,
                       const AlgoMuonPtr& procMuon,
                       const TrackingParticle& trackingParticle,
                       TrajectoryStateOnSurface& tsof);

  std::vector<MatchingResult> cleanMatching(std::vector<MatchingResult> matchingResults,
                                            std::vector<const l1t::RegionalMuonCand*>& muonCands,
                                            AlgoMuons& ghostBustedProcMuons);

  std::vector<MatchingResult> match(std::vector<const l1t::RegionalMuonCand*>& muonCands,
                                    AlgoMuons& ghostBustedProcMuons,
                                    const edm::SimTrackContainer* simTracks,
                                    const edm::SimVertexContainer* simVertices,
                                    std::function<bool(const SimTrack&)> const& simTrackFilter);

  std::vector<MatchingResult> match(std::vector<const l1t::RegionalMuonCand*>& muonCands,
                                    AlgoMuons& ghostBustedProcMuons,
                                    const TrackingParticleCollection* trackingParticles,
                                    std::function<bool(const TrackingParticle&)> const& simTrackFilter);

  //matching without any propagation, just checking basic geometrical agreement between simMuon and candidates
  //problem with propagation is the it does not work for low pt muons (pt < ~3GeV)
  //which is not good for dumping the data for the NN training. So for that purpose it is better to use the matchSimple
  std::vector<MatchingResult> matchSimple(std::vector<const l1t::RegionalMuonCand*>& muonCands,
                                          AlgoMuons& ghostBustedProcMuons,
                                          const edm::SimTrackContainer* simTracks,
                                          const edm::SimVertexContainer* simVertices,
                                          std::function<bool(const SimTrack&)> const& simTrackFilter);

  std::vector<MatchingResult> getMatchingResults() { return matchingResults; }

private:
  const OMTFConfiguration* omtfConfig;

  const edm::ParameterSet& edmCfg;

  AlgoMuons gbCandidates;
  std::vector<MatchingResult> matchingResults;

  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord>& magneticFieldEsToken;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord>& propagatorEsToken;

  edm::ESHandle<MagneticField> magField;
  edm::ESHandle<Propagator> propagator;

  TH1D* deltaPhiPropCandMean = nullptr;
  TH1D* deltaPhiPropCandStdDev = nullptr;

  bool usePropagation = false;
};

#endif /* L1T_OmtfP1_TOOLS_MUONCANDIDATEMATCHER_H_ */
