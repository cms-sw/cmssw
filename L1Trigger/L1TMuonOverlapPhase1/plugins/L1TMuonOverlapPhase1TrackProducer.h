#ifndef L1T_OmtfP1_L1TMuonOverlapPhase1TrackProducer_H
#define L1T_OmtfP1_L1TMuonOverlapPhase1TrackProducer_H

#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OMTFReconstruction.h"

#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/one/EDProducer.h"

#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"

/**
 * The OMTF emulator cannot be run in multiple threads, because GoldenPatternBase keeps GoldenPatternResult's
 * which are then used in the GoldenPatternBase::finalise called at the end of OMTFProcessor<GoldenPatternType>::processInput.
 * Also patterns generation relies on updating the statistics for the patterns
 * This can be fixed, but requires some work.
 */
class L1TMuonOverlapPhase1TrackProducer : public edm::one::EDProducer<edm::one::WatchRuns> {
public:
  L1TMuonOverlapPhase1TrackProducer(const edm::ParameterSet&);

  ~L1TMuonOverlapPhase1TrackProducer() override;

  void beginJob() override;

  void endJob() override;

  void beginRun(edm::Run const& run, edm::EventSetup const& iSetup) override;

  void endRun(edm::Run const& run, edm::EventSetup const& iSetup) override{};

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  //edm::EDGetTokenT<edm::SimTrackContainer> inputTokenSimHit;  //TODO remove

  MuStubsInputTokens muStubsInputTokens;

  edm::ESGetToken<L1TMuonOverlapParams, L1TMuonOverlapParamsRcd> omtfParamsEsToken;

  //needed for AngleConverterBase
  MuonGeometryTokens muonGeometryTokens;

  ///needed by tools/CandidateSimMuonMatcher.h
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magneticFieldEsToken;
  edm::ESGetToken<Propagator, TrackingComponentsRecord> propagatorEsToken;

  OMTFReconstruction omtfReconstruction;
};

#endif
