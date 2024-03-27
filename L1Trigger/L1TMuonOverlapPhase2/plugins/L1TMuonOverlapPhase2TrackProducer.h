#ifndef OMTFProducer_H
#define OMTFProducer_H

#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "L1Trigger/L1TMuonOverlapPhase2/interface/OmtfEmulation.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"

class L1TMuonOverlapPhase2TrackProducer : public edm::one::EDProducer<edm::one::WatchRuns> {
public:
  L1TMuonOverlapPhase2TrackProducer(const edm::ParameterSet&);

  ~L1TMuonOverlapPhase2TrackProducer() override = default;

  void beginJob() override;

  void endJob() override;

  void beginRun(edm::Run const& run, edm::EventSetup const& iSetup) override;

  void endRun(edm::Run const& run, edm::EventSetup const& iSetup) override{};

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  MuStubsInputTokens muStubsInputTokens;

  edm::ESGetToken<L1TMuonOverlapParams, L1TMuonOverlapParamsRcd> omtfParamsEsToken;

  //needed for AngleConverterBase
  MuonGeometryTokens muonGeometryTokens;

  ///needed by tools/CandidateSimMuonMatcher.h
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magneticFieldEsToken;
  edm::ESGetToken<Propagator, TrackingComponentsRecord> propagatorEsToken;

  OmtfEmulation omtfEmulation;
};

#endif
