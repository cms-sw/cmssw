/*
 * OmtfEmulation.cpp
 *
 *  Created on: May 20, 2020
 *      Author: kbunkow
 */

#include <memory>

#include "L1Trigger/L1TMuonOverlapPhase2/interface/OmtfEmulation.h"
#include "L1Trigger/L1TMuonOverlapPhase2/interface/InputMakerPhase2.h"

#include "DataFormats/L1TMuonPhase2/interface/Constants.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <memory>

OmtfEmulation::OmtfEmulation(const edm::ParameterSet& edmParameterSet,
                             MuStubsInputTokens& muStubsInputTokens,
                             MuStubsPhase2InputTokens& muStubsPhase2InputTokens)
    : OMTFReconstruction(edmParameterSet, muStubsInputTokens), muStubsPhase2InputTokens(muStubsPhase2InputTokens), omtfProcPhase2(omtfConfig.get(), omtfProc) {}

void OmtfEmulation::beginJob() {
  if (edmParameterSet.exists("usePhase2DTPrimitives") && edmParameterSet.getParameter<bool>("usePhase2DTPrimitives")) {
    inputMaker = std::make_unique<InputMakerPhase2>(edmParameterSet,
                                                    muStubsInputTokens,
                                                    muStubsPhase2InputTokens,
                                                    omtfConfig.get(),
                                                    std::make_unique<OmtfPhase2AngleConverter>());
  } else {
    inputMaker = std::make_unique<OMTFinputMaker>(
        edmParameterSet, muStubsInputTokens, omtfConfig.get(), std::make_unique<OmtfAngleConverter>());
  }
}

void OmtfEmulation::beginRun(edm::Run const& run,
                                  edm::EventSetup const& eventSetup,
                                  edm::ESGetToken<L1TMuonOverlapParams, L1TMuonOverlapParamsRcd>& omtfParamsEsToken,
                                  const MuonGeometryTokens& muonGeometryTokens,
                                  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord>& magneticFieldEsToken,
                                  const edm::ESGetToken<Propagator, TrackingComponentsRecord>& propagatorEsToken) {

  OMTFReconstruction::beginRun(run, eventSetup, omtfParamsEsToken, muonGeometryTokens, magneticFieldEsToken, propagatorEsToken);

  omtfProcPhase2.beginRun(edmParameterSet, eventSetup);
}

OmtfEmulation::OmtfOutptuCollections OmtfEmulation::run(
    const edm::Event& iEvent,
    const edm::EventSetup& evSetup) {
  LogTrace("l1tOmtfEventPrint") << "\n" << __FUNCTION__ << ":" << __LINE__ << " iEvent " << iEvent.id().event() << endl;
  inputMaker->loadAndFilterDigis(iEvent);

  for (auto& obs : observers) {
    obs->observeEventBegin(iEvent);
  }

  OmtfOutptuCollections outptuCollections;
  outptuCollections.constrSaMuons = std::make_unique<l1t::SAMuonCollection>();
  outptuCollections.unConstrSaMuons = std::make_unique<l1t::SAMuonCollection>();
  outptuCollections.regionalCandidates = std::make_unique<l1t::RegionalMuonCandBxCollection>();
  outptuCollections.regionalCandidates->setBXRange(bxMin, bxMax);

  FinalMuons allFinalMuons;

  ///The order is important: first put omtf_pos candidates, then omtf_neg.
  for (int bx = bxMin; bx <= bxMax; bx++) {
    for(unsigned int iSide = 0; iSide < 2; ++iSide) {
      l1t::tftype mtfType = (iSide == 0) ? l1t::tftype::omtf_pos : l1t::tftype::omtf_neg;
      for (unsigned int iProcessor = 0; iProcessor < omtfConfig->nProcessors(); ++iProcessor) {
        FinalMuons finalMuons = omtfProcPhase2.run(iProcessor, mtfType, bx, inputMaker.get(), observers);

        l1t::SAMuonCollection constrSAMuons = omtfProcPhase2.getSAMuons(iProcessor, mtfType, finalMuons, true);
        for (auto& saMuon : constrSAMuons) {
          outptuCollections.constrSaMuons->push_back(saMuon);
        }

        l1t::SAMuonCollection unconstrSAMuons = omtfProcPhase2.getSAMuons(iProcessor, mtfType, finalMuons, false);
        for (auto& saMuon : unconstrSAMuons) {
          outptuCollections.unConstrSaMuons->push_back(saMuon);
        }

        //getRegionalMuonCands calls convertToGmtScalesPhase1, it sets value ptGmt, phiGmt, etaGmt in finalMuons
        //so that regionalCandidates have values in phase-1 scales
        //but it will affect the finalMuons stored in allFinalMuons as they are shared pointers,
        //Therefore convertToGmtScalesPhase2 is called again.
        std::vector<l1t::RegionalMuonCand> candMuons = omtfProc->getRegionalMuonCands(iProcessor, mtfType, finalMuons);
        for (auto& candMuon : candMuons) {
          outptuCollections.regionalCandidates->push_back(bx, candMuon);
        }
        //TODO remove it when RegionalCandidates are not needed anymore
        for (auto& finalMuon : finalMuons) {
          omtfProcPhase2.convertToGmtScalesPhase2(iProcessor, mtfType, finalMuon);
        }

        allFinalMuons.insert(allFinalMuons.end(), finalMuons.begin(), finalMuons.end());
      }
    }

    //edm::LogInfo("OMTFReconstruction") <<"OMTF:  Number of candidates in BX="<<bx<<": "<<candidates->size(bx) << std::endl;;
  }

  for (auto& obs : observers) {
    obs->observeEventEnd(iEvent, allFinalMuons);
  }

  return outptuCollections;
}
