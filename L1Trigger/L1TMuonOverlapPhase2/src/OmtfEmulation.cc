/*
 * OmtfEmulation.cpp
 *
 *  Created on: May 20, 2020
 *      Author: kbunkow
 */

#include <memory>

#include "L1Trigger/L1TMuonOverlapPhase2/interface/OmtfEmulation.h"
#include "L1Trigger/L1TMuonOverlapPhase2/interface/InputMakerPhase2.h"
#include "L1Trigger/L1TMuonOverlapPhase2/interface/PtAssignmentNNRegression.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <memory>

OmtfEmulation::OmtfEmulation(const edm::ParameterSet& edmParameterSet,
                             MuStubsInputTokens& muStubsInputTokens,
                             edm::EDGetTokenT<L1Phase2MuDTPhContainer> inputTokenDTPhPhase2)
    : OMTFReconstruction(edmParameterSet, muStubsInputTokens), inputTokenDTPhPhase2(inputTokenDTPhPhase2) {}

OmtfEmulation::~OmtfEmulation() {}

void OmtfEmulation::beginJob() {
  if (edmParameterSet.exists("usePhase2DTPrimitives") && edmParameterSet.getParameter<bool>("usePhase2DTPrimitives")) {
    inputMaker = std::make_unique<InputMakerPhase2>(edmParameterSet,
                                                    muStubsInputTokens,
                                                    inputTokenDTPhPhase2,
                                                    omtfConfig.get(),
                                                    std::make_unique<OmtfPhase2AngleConverter>());
  } else {
    inputMaker = std::make_unique<OMTFinputMaker>(
        edmParameterSet, muStubsInputTokens, omtfConfig.get(), std::make_unique<OmtfAngleConverter>());
  }
}

void OmtfEmulation::addObservers(const MuonGeometryTokens& muonGeometryTokens,
                                 const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord>& magneticFieldEsToken,
                                 const edm::ESGetToken<Propagator, TrackingComponentsRecord>& propagatorEsToken) {
  if (observers.empty()) {  //assuring it is done only at the first run
    OMTFReconstruction::addObservers(muonGeometryTokens, magneticFieldEsToken, propagatorEsToken);
    /*    if(edmParameterSet.exists("patternsPtAssignment") && edmParameterSet.getParameter<bool>("patternsPtAssignment")) {
      //std::string rootFileName = edmParameterSet.getParameter<std::string>("dumpHitsFileName");
      .emplace_back(std::make_unique<PatternsPtAssignment>(edmParameterSet, omtfConfig.get(), omtfProcGoldenPat->getPatterns(), ""));
    }*/
  }

  if (edmParameterSet.exists("neuralNetworkFile") && !ptAssignment) {
    edm::LogImportant("OMTFReconstruction") << "constructing PtAssignmentNNRegression" << std::endl;
    std::string neuralNetworkFile = edmParameterSet.getParameter<edm::FileInPath>("neuralNetworkFile").fullPath();
    ptAssignment = std::make_unique<PtAssignmentNNRegression>(edmParameterSet, omtfConfig.get(), neuralNetworkFile);
  }

  auto omtfProcGoldenPat = dynamic_cast<OMTFProcessor<GoldenPattern>*>(omtfProc.get());
  if (omtfProcGoldenPat) {
    omtfProcGoldenPat->setPtAssignment(ptAssignment.get());
    //omtfProcGoldenPat can be constructed from scratch each run, so ptAssignment is set herer every run
  }
}
