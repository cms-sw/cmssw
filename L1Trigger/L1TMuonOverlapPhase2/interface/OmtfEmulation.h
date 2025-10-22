/*
 * OmtfEmulation.h
 *
 *  Created on: May 20, 2020
 *      Author: kbunkow
 */

#ifndef L1Trigger_L1TMuonOverlapPhase2_OmtfEmulation_h
#define L1Trigger_L1TMuonOverlapPhase2_OmtfEmulation_h

#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTPhContainer.h"
#include "DataFormats/L1TMuonPhase2/interface/SAMuon.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OMTFReconstruction.h"
#include "L1Trigger/L1TMuonOverlapPhase2/interface/OmtfPhase2AngleConverter.h"
#include "L1Trigger/L1TMuonOverlapPhase2/interface/InputMakerPhase2.h"

class OmtfEmulation : public OMTFReconstruction {
public:
  OmtfEmulation(const edm::ParameterSet& edmParameterSet,
                MuStubsInputTokens& muStubsInputTokens,
                MuStubsPhase2InputTokens& muStubsPhase2InputTokens);

  void beginJob();

  ~OmtfEmulation() override = default;

  void addObservers(const MuonGeometryTokens& muonGeometryTokens,
                    const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord>& magneticFieldEsToken,
                    const edm::ESGetToken<Propagator, TrackingComponentsRecord>& propagatorEsToken) override;

  //RegionalMuonCandBxCollection is filled for backward compatibility of analyzers etc.

  struct OmtfOutptuCollections {
    std::unique_ptr<l1t::SAMuonCollection> constrSaMuons; //ip constrained candidates
    std::unique_ptr<l1t::SAMuonCollection> unConstrSaMuons; //ip unconstrained candidates
    std::unique_ptr<l1t::RegionalMuonCandBxCollection> regionalCandidates; //for backward compatibility of analyzers etc.
  };

  OmtfOutptuCollections run(const edm::Event& iEvent,
                                             const edm::EventSetup& evSetup);

private:
  MuStubsPhase2InputTokens& muStubsPhase2InputTokens;
  unique_ptr<PtAssignmentBase> ptAssignment;

  std::map<unsigned int, int> firedLayersToQuality;

  void assignQualityPhase2(AlgoMuons::value_type& algoMuon);

  void convertToGmtScalesPhase2(unsigned int iProcessor, l1t::tftype mtfType, FinalMuonPtr& finalMuon);

  l1t::SAMuonCollection getSAMuons(unsigned int iProcessor,
                                   l1t::tftype mtfType,
                                   FinalMuons& finalMuons,
                                   bool costrainedPt);
};

#endif /* L1Trigger_L1TMuonOverlapPhase2_OmtfEmulation_h */
