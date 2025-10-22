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

#include "DataFormats/L1TMuonPhase2/interface/Constants.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <memory>

OmtfEmulation::OmtfEmulation(const edm::ParameterSet& edmParameterSet,
                             MuStubsInputTokens& muStubsInputTokens,
                             MuStubsPhase2InputTokens& muStubsPhase2InputTokens)
    : OMTFReconstruction(edmParameterSet, muStubsInputTokens), muStubsPhase2InputTokens(muStubsPhase2InputTokens) {}

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

  //.....................rrrrrrrrccccdddddd
  //.....................765432109876543210
  firedLayersToQuality[0b000000110000000011] = 1;
  firedLayersToQuality[0b000000100000000011] = 1;
  firedLayersToQuality[0b000000010000000011] = 1;
  firedLayersToQuality[0b000000110000000001] = 1;
  firedLayersToQuality[0b000001000000001100] = 1;
  firedLayersToQuality[0b000011000000001100] = 1;
  firedLayersToQuality[0b000010000000001100] = 1;
  firedLayersToQuality[0b000011000000000100] = 1;
  firedLayersToQuality[0b000000011000000001] = 1;
  firedLayersToQuality[0b001000010000000001] = 1;

  firedLayersToQuality[0b000100000000110000] = 1;
  firedLayersToQuality[0b001100000000010000] = 1;

  firedLayersToQuality[0b010000110000000001] = 8;
  firedLayersToQuality[0b000000111110000001] = 8;
  firedLayersToQuality[0b000000001000000011] = 8;
  firedLayersToQuality[0b000000111000000001] = 8;
  firedLayersToQuality[0b000000101000000001] = 8;
  firedLayersToQuality[0b010000011000000001] = 8;
  firedLayersToQuality[0b010000100000000001] = 8;
  firedLayersToQuality[0b000000110100000001] = 8;
  firedLayersToQuality[0b000000100100000001] = 8;
  firedLayersToQuality[0b001000100000000001] = 8;
  firedLayersToQuality[0b010000010000000001] = 8;
  firedLayersToQuality[0b001000110000000001] = 8;
  firedLayersToQuality[0b001000110000000000] = 8;
  firedLayersToQuality[0b000000010100000001] = 8;
  firedLayersToQuality[0b000010100000000001] = 8;
  firedLayersToQuality[0b000000100010000001] = 8;
  firedLayersToQuality[0b001010010000000101] = 8;
  firedLayersToQuality[0b100000000000000011] = 8;
  firedLayersToQuality[0b011011000000000000] = 8;
  firedLayersToQuality[0b000010110000000001] = 8;
  firedLayersToQuality[0b001001110000000001] = 8;
  firedLayersToQuality[0b000010100000000101] = 8;
  firedLayersToQuality[0b000011110000000001] = 8;
  firedLayersToQuality[0b000011110000001101] = 8;
  firedLayersToQuality[0b000011100000000101] = 8;
  firedLayersToQuality[0b000011110000000101] = 8;
  firedLayersToQuality[0b000100000001110000] = 8;
  firedLayersToQuality[0b000001110000001101] = 8;
  firedLayersToQuality[0b000000110110000001] = 8;
  firedLayersToQuality[0b000001110000000001] = 8;
  firedLayersToQuality[0b001000010001000001] = 8;
  firedLayersToQuality[0b000001100000000101] = 8;
  firedLayersToQuality[0b000001100000000001] = 8;
  firedLayersToQuality[0b000001110000000101] = 8;
  firedLayersToQuality[0b001001110001000001] = 8;
  firedLayersToQuality[0b000010110000000101] = 8;
  firedLayersToQuality[0b000000010001000001] = 8;
  firedLayersToQuality[0b000000100110000001] = 8;
  firedLayersToQuality[0b001001100000001100] = 8;
  firedLayersToQuality[0b000001010000000001] = 8;
  firedLayersToQuality[0b000010100000000011] = 8;
  firedLayersToQuality[0b000000100001000001] = 8;
  firedLayersToQuality[0b001000110001000001] = 8;
  firedLayersToQuality[0b000000010010000001] = 8;
  firedLayersToQuality[0b000001010000000101] = 8;
  firedLayersToQuality[0b100000100110000001] = 8;
  firedLayersToQuality[0b000010010000000101] = 8;
  firedLayersToQuality[0b000000110010000001] = 8;
  firedLayersToQuality[0b000000000000110100] = 8;
  firedLayersToQuality[0b000000010000000101] = 8;
  firedLayersToQuality[0b000000110001000001] = 8;
  firedLayersToQuality[0b000000010000001100] = 8;
  firedLayersToQuality[0b000010110000001101] = 8;
  firedLayersToQuality[0b000011010000001101] = 8;
  firedLayersToQuality[0b000000100000010001] = 8;
  firedLayersToQuality[0b000000110000000101] = 8;
  firedLayersToQuality[0b000001100000000111] = 8;
  firedLayersToQuality[0b000000100000000101] = 8;
  firedLayersToQuality[0b010000010010000001] = 8;
  firedLayersToQuality[0b000001100000001101] = 8;
  firedLayersToQuality[0b000011100000000111] = 8;
  firedLayersToQuality[0b000000010110000001] = 8;
  firedLayersToQuality[0b000011110000000111] = 8;
  firedLayersToQuality[0b000000011100000000] = 8;
  firedLayersToQuality[0b001000010000000011] = 8;
  firedLayersToQuality[0b000001110000000011] = 8;
  firedLayersToQuality[0b000100000000110000] = 8;
  firedLayersToQuality[0b000111100000110100] = 8;
  firedLayersToQuality[0b010000010010000000] = 8;
  firedLayersToQuality[0b100000010100000000] = 8;
  firedLayersToQuality[0b001000100000000011] = 8;
  firedLayersToQuality[0b000011100000001101] = 8;
  firedLayersToQuality[0b100000011100000000] = 8;
  firedLayersToQuality[0b110000011110000001] = 8;
  //firedLayersToQuality[0b000000000000110011] = 8;
  //firedLayersToQuality[0b000000100110000011] = 8;
  //firedLayersToQuality[0b110000000100000000] = 8;
  //firedLayersToQuality[0b001011110001001101] = 8;
  //firedLayersToQuality[0b010000100001000011] = 8;
  //firedLayersToQuality[0b000001100000001100] = 8;
  //firedLayersToQuality[0b000001110001000011] = 8;
  //firedLayersToQuality[0b011000000010000000] = 8;
  //firedLayersToQuality[0b001000110100000011] = 8;
  //firedLayersToQuality[0b010001000011000000] = 8;
  //firedLayersToQuality[0b100000000110000000] = 8;
  //firedLayersToQuality[0b000000000000111100] = 8;
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

  //addObservers is called in OMTFReconstruction::beginRun after the omtfProc is constructed, therefore here we can ptAssignment in omtfProc
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

  omtfProc->setAssignQualityFunction(
      [&](AlgoMuons::value_type& algoMuon) { this->assignQualityPhase2(algoMuon); });
}

void OmtfEmulation::assignQualityPhase2(AlgoMuons::value_type& algoMuon) {
  if (abs(algoMuon->getEtaHw()) >= 121) { //TODO take into account the eta scale
    algoMuon->setQuality(0); // changed from 4 on request from HI
    return;
  }

  auto it = firedLayersToQuality.find(algoMuon->getFiredLayerBits());
  if (algoMuon->getPtConstr() == 0) {
    algoMuon->setQuality(0);  //default value
    return;
  }
  
  if (it != firedLayersToQuality.end()) {
    algoMuon->setQuality(it->second);
  } else {
    algoMuon->setQuality(12);  //default value
  }
};

void OmtfEmulation::convertToGmtScalesPhase2(unsigned int iProcessor, l1t::tftype mtfType, FinalMuonPtr& finalMuon) {
  //ptAssignment (NN) is used only if there was valid candidate from pattern logic
  //it overrides the pt from the pattern logic
  if (ptAssignment) {
    //PtNNConstr should be in GeV
    finalMuon->setPtGev(finalMuon->getAlgoMuon()->getPtNNConstr());

    //TODO use getPtNNUnconstr when the network with upt is trained to set setPtUnconstrGev()
  }

  //in getFinalMuons the PtGeV is set to 0 in this case, as it is like that for the phase-1.
  //but it is better to set non-0 pt in this case, so we set 1 GeV
  if (finalMuon->getAlgoMuon()->getPdfSumConstr() == 0 && finalMuon->getAlgoMuon()->getPtUnconstr() > 0)
    finalMuon->setPtGev(1.0); //set to 1 GeV to be able to distinguish from pt=0, which means no candidate

  int maxPtHw = (1 << Phase2L1GMT::BITSPT) - 1;

  int ptHwConstr = (finalMuon->getPtGev() * (1. / Phase2L1GMT::LSBpt));

  if (ptHwConstr >= maxPtHw)
    ptHwConstr = maxPtHw;

  finalMuon->setPtGmt(ptHwConstr);


  int ptHwUnConstr = finalMuon->getPtUnconstrGev() * (1. / Phase2L1GMT::LSBpt);

  if (ptHwUnConstr >= maxPtHw)
    ptHwUnConstr = maxPtHw;

  finalMuon->setPtUnconstrGmt(ptHwUnConstr);


  if (mtfType == l1t::omtf_pos) {
    finalMuon->setEtaGmt(finalMuon->getAlgoMuon()->getEtaHw());
  }
  else {
    finalMuon->setEtaGmt((-1) * finalMuon->getAlgoMuon()->getEtaHw());
  }

  int globPhi = omtfConfig->procPhiOmtfToGlobalPhiOmtf(iProcessor, finalMuon->getAlgoMuon()->getPhi());
  int gmtPhiBins = 1 << Phase2L1GMT::BITSPHI;
  int omtfToGmtFactorPhi = std::lround(gmtPhiBins * (1 << 12) / double(omtfConfig->nPhiBins()) ) ;
  int gmtPhi = (globPhi * omtfToGmtFactorPhi) >> 12;
  finalMuon->setPhiGmt(gmtPhi);

  int omtfToGmtFactorEta = std::lround(omtfConfig->etaUnit()  * (1 << 12) / Phase2L1GMT::LSBeta ) ;
  int gmtEta = (finalMuon->getAlgoMuon()->getEtaHw() * omtfToGmtFactorEta) >> 12;
  if (mtfType == l1t::omtf_neg)
    gmtEta = -gmtEta;
  finalMuon->setEtaGmt(gmtEta);

  //finalMuon.setHwSignValid(1);
}

l1t::SAMuonCollection OmtfEmulation::getSAMuons(unsigned int iProcessor,
                                                l1t::tftype mtfType,
                                                FinalMuons& finalMuons,
                                                bool costrainedPt) {
  l1t::SAMuonCollection saMuons;
  for (auto& finalMuon : finalMuons) {
    convertToGmtScalesPhase2(iProcessor, mtfType, finalMuon);

    int charge = finalMuon->getSign();

    unsigned int pt = costrainedPt ? finalMuon->getPtGmt() : finalMuon->getPtUnconstrGmt();

    LogTrace("OMTFReconstruction") << "OmtfEmulation::getSAMuons finalMuon->getPtGmt(): "
        << finalMuon->getPtGmt() << " finalMuon->getPtUnconstrGmt() "<< finalMuon->getPtUnconstrGmt() << std::endl;

    int phi = finalMuon->getPhiGmt(); 
    int eta = finalMuon->getEtaGmt();  

    int z0 = 0;  
    // Use 2 bits with LSB = 30cm for BMTF and 25cm for EMTF currently, but subjet to change
    int d0 = 0; //finalMuon->getHwD0();

    unsigned int qual = finalMuon->getQuality();

    //TODO FIX
    //Here do not use the word format to GT but use the word format expected by GMT
    /*
    int bstart = 0;
    wordtype word(0);
    bstart = wordconcat<wordtype>(word, bstart, 1, 1);
    bstart = wordconcat<wordtype>(word, bstart, charge, 1);
    bstart = wordconcat<wordtype>(word, bstart, pt, BITSPT);
    bstart = wordconcat<wordtype>(word, bstart, phi, BITSPHI);
    bstart = wordconcat<wordtype>(word, bstart, eta, BITSETA);
    //  bstart = wordconcat<wordtype>(word, bstart, z0, BITSSAZ0); NOT YET SUPPORTED BY GMT
    bstart = wordconcat<wordtype>(word, bstart, d0, BITSSAD0);
    bstart = wordconcat<wordtype>(
        word, bstart, qual, 8);  //FOR NOW 8 bits to be efficienct with Ghost busting. THIS IS ***NOT*** THE FINAL QUALITY
*/

    // Calculate Lorentz Vector
	//TODO for the vertex constrained muon, the z0 and d0 by definition should be 0 - then why give it? 
    math::PtEtaPhiMLorentzVector p4Constr(pt * Phase2L1GMT::LSBpt, eta * Phase2L1GMT::LSBeta, phi * Phase2L1GMT::LSBphi, 0.0);
    l1t::SAMuon saMuonConstr(p4Constr, charge, pt, eta, phi, z0, d0, qual);
    saMuonConstr.setTF(mtfType);
    //samuon.setWord(word);

    if (saMuonConstr.hwPt() > 0) {
      saMuons.push_back(saMuonConstr);
    }
  }

  return saMuons;
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
        FinalMuons finalMuons = omtfProc->run(iProcessor, mtfType, bx, inputMaker.get(), observers);

        //getRegionalMuonCands calls convertToGmtScalesPhase1, it sets value eta, phi, pt finalMuons
        //so regionalCandidates have values in phase-1 scales
        std::vector<l1t::RegionalMuonCand> candMuons =
            omtfProc->getRegionalMuonCands(iProcessor, mtfType, finalMuons);
        for (auto& candMuon : candMuons) {
          outptuCollections.regionalCandidates->push_back(bx, candMuon);
        }

        for (auto& finalMuon : finalMuons) {
          convertToGmtScalesPhase2(iProcessor, mtfType, finalMuon);
        }

        l1t::SAMuonCollection constrSAMuons = getSAMuons(iProcessor, mtfType, finalMuons, true);
        for (auto& saMuon : constrSAMuons) {
          outptuCollections.unConstrSaMuons->push_back(saMuon);
        }

        l1t::SAMuonCollection unconstrSAMuons = getSAMuons(iProcessor, mtfType, finalMuons, false);
        for (auto& saMuon : unconstrSAMuons) {
          outptuCollections.unConstrSaMuons->push_back(saMuon);
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
