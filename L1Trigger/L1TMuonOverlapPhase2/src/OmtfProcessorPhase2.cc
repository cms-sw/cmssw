/*
 * OmtfProcessorPhase2.cc
 *
 *  Created on: Oct 29, 2025
 *      Author: kbunkow
 */

#include <L1Trigger/L1TMuonOverlapPhase2/interface/NNRegression.h>
#include "L1Trigger/L1TMuonOverlapPhase2/interface/OmtfProcessorPhase2.h"

OmtfProcessorPhase2::OmtfProcessorPhase2(const OMTFConfiguration* omtfConfig,
                                         const unique_ptr<IProcessorEmulator>& omtfProc)
    : omtfConfig(omtfConfig), omtfProc(omtfProc) {
  //TODO read from configuration
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

OmtfProcessorPhase2::~OmtfProcessorPhase2() {}

void OmtfProcessorPhase2::beginRun(const edm::ParameterSet& edmParameterSet, edm::EventSetup const& iSetup) {
  if (edmParameterSet.exists("neuralNetworkFile") && !mlModel) {
    edm::LogImportant("OMTFReconstruction") << "constructing PtAssignmentNNRegression" << std::endl;
    std::string neuralNetworkFile = edmParameterSet.getParameter<edm::FileInPath>("neuralNetworkFile").fullPath();
    mlModel = std::make_unique<NNRegression>(edmParameterSet, omtfConfig, neuralNetworkFile);
  }
}

void OmtfProcessorPhase2::assignQualityPhase2(AlgoMuons::value_type& algoMuon) {
  if (!algoMuon->isValid())
    return;

  //better not to assume anything about quality for pt=0 candidates
  /*if (algoMuon->getPtConstr() == 0) {
    algoMuon->setQuality(0);  //default value
    return;
  }*/

  //TODO agree on meaning of quality = 0
  if (abs(algoMuon->getEtaHw()) >= omtfConfig->etaToHwEta(1.3)) {
    algoMuon->setQuality(0);
    return;
  }

  LogTrace("OMTFReconstruction") << "OmtfEmulation::assignQualityPhase2 algoMuon->getFiredLayerBits() "
                                 << std::bitset<18>(algoMuon->getFiredLayerBits()) << " algoMuon->getEtaHw() "
                                 << algoMuon->getEtaHw() << " omtfConfig->etaToHwEta(1.25) "
                                 << omtfConfig->etaToHwEta(1.25) << std::endl;

  if (abs(algoMuon->getEtaHw()) >= omtfConfig->etaToHwEta(1.25) &&
      (algoMuon->getFiredLayerBits() == std::bitset<18>("100000001110000000").to_ulong() ||
       algoMuon->getFiredLayerBits() == std::bitset<18>("000000001110000000").to_ulong() ||
       algoMuon->getFiredLayerBits() == std::bitset<18>("100000000110000000").to_ulong() ||
       algoMuon->getFiredLayerBits() == std::bitset<18>("100000001100000000").to_ulong() ||
       algoMuon->getFiredLayerBits() == std::bitset<18>("100000001010000000").to_ulong())) {
    algoMuon->setQuality(1);
    LogTrace("OMTFReconstruction") << "OmtfEmulation::assignQualityPhase2 assigned quality 1 for etaHw "
                                   << algoMuon->getEtaHw() << std::endl;
    return;
  }

  auto it = firedLayersToQuality.find(algoMuon->getFiredLayerBits());
  if (it != firedLayersToQuality.end()) {
    algoMuon->setQuality(it->second);
  } else {
    algoMuon->setQuality(12);  //default value
  }
};

void OmtfProcessorPhase2::convertToGmtScalesPhase2(unsigned int iProcessor,
                                                   l1t::tftype mtfType,
                                                   FinalMuonPtr& finalMuon) {
  //ptAssignment (NN) is used only if there was valid candidate from pattern logic
  //it overrides the pt from the pattern logic
  if (mlModel) {
    //PtNNConstr should be in GeV
    finalMuon->setPtGev(finalMuon->getAlgoMuon()->getPtNNConstr());
    finalMuon->setSign(finalMuon->getAlgoMuon()->getChargeNNConstr() < 0 ? 1 : 0);

    //TODO use getPtNNUnconstr when the network with upt is trained to set setPtUnconstrGev()
    LogTrace("OMTFReconstruction") << "OmtfEmulation::convertToGmtScalesPhase2 using Pt from NN "
                                   << " iProcessor " << iProcessor << " ptNN "
                                   << finalMuon->getAlgoMuon()->getPtNNConstr() << " ptPatterns "
                                   << finalMuon->getAlgoMuon()->getPtConstr() << " ChargeNN "
                                   << finalMuon->getAlgoMuon()->getChargeNNConstr() << " Charge "
                                   << finalMuon->getAlgoMuon()->getChargeConstr() << std::endl;
  }

  //in getFinalMuons the PtGeV is set to 0 in this case, as it is like that for the phase-1.
  //but for the phase-2 pt = 0 means empty candidate, so we set 1 GeV in this case
  if (finalMuon->getAlgoMuon()->getPdfSumConstr() == 0 && finalMuon->getAlgoMuon()->getPtUnconstr() > 0)
    finalMuon->setPtGev(1.0);  //set to 1 GeV to be able to distinguish from pt=0, which means no candidate

  int maxPtHw = (1 << Phase2L1GMT::BITSPT) - 1;

  int ptHwConstr = finalMuon->getPtGev() * (1. / Phase2L1GMT::LSBpt);

  if (ptHwConstr >= maxPtHw)
    ptHwConstr = maxPtHw;

  finalMuon->setPtGmt(ptHwConstr);

  int ptHwUnConstr = finalMuon->getPtUnconstrGev() * (1. / Phase2L1GMT::LSBpt);

  if (ptHwUnConstr >= maxPtHw)
    ptHwUnConstr = maxPtHw;

  finalMuon->setPtUnconstrGmt(ptHwUnConstr);

  LogTrace("OMTFReconstruction") << "convertToGmtScalesPhase2 finalMuon->getPtGev() iProcessor " << iProcessor
                                 << " PtGev " << finalMuon->getPtGev() << " PtUnconstrGev "
                                 << finalMuon->getPtUnconstrGev() << std::endl;

  if (mtfType == l1t::omtf_pos) {
    finalMuon->setEtaGmt(finalMuon->getAlgoMuon()->getEtaHw());
  } else {
    finalMuon->setEtaGmt((-1) * finalMuon->getAlgoMuon()->getEtaHw());
  }

  int globPhi = omtfConfig->procPhiOmtfToGlobalPhiOmtf(iProcessor, finalMuon->getAlgoMuon()->getPhi());
  int gmtPhiBins = 1 << Phase2L1GMT::BITSPHI;
  int omtfToGmtFactorPhi = std::lround(gmtPhiBins * (1 << 12) / double(omtfConfig->nPhiBins()));
  int gmtPhi = (globPhi * omtfToGmtFactorPhi) >> 12;
  finalMuon->setPhiGmt(gmtPhi);

  static const int omtfToGmtFactorEta = std::lround(omtfConfig->etaUnit() / Phase2L1GMT::LSBeta);  //should be 2
  LogTrace("OMTFReconstruction") << "OmtfEmulation::convertToGmtScalesPhase2 omtfToGmtFactorEta " << omtfToGmtFactorEta
                                 << std::endl;
  int gmtEta = (finalMuon->getAlgoMuon()->getEtaHw() * omtfToGmtFactorEta);
  if (mtfType == l1t::omtf_neg)
    gmtEta = -gmtEta;
  finalMuon->setEtaGmt(gmtEta);

  //finalMuon.setHwSignValid(1);
}

l1t::SAMuonCollection OmtfProcessorPhase2::getSAMuons(unsigned int iProcessor,
                                                      l1t::tftype mtfType,
                                                      FinalMuons& finalMuons,
                                                      bool costrainedPt) {
  l1t::SAMuonCollection saMuons;
  for (auto& finalMuon : finalMuons) {
    int charge = finalMuon->getSign();

    unsigned int pt = costrainedPt ? finalMuon->getPtGmt() : finalMuon->getPtUnconstrGmt();
    int d0 = costrainedPt ? 0 : 50 / Phase2L1GMT::LSBSAd0;  //finalMuon->getHwD0();
    if (costrainedPt == false) {
      //this assures the collection of constrained and unconstrained muons have the same size
      //muons that are not displaced also should be in the unconstrained collection
      if (finalMuon->getPtUnconstrGmt() == 0) {
        pt = finalMuon->getPtGmt();
        d0 = 0;
      }
    }

    LogTrace("OMTFReconstruction") << "OmtfEmulation::getSAMuons finalMuon->getPtGmt(): " << finalMuon->getPtGmt()
                                   << " finalMuon->getPtUnconstrGmt() " << finalMuon->getPtUnconstrGmt() << std::endl;

    int phi = finalMuon->getPhiGmt();
    int eta = finalMuon->getEtaGmt();

    int z0 = 0;

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
    math::PtEtaPhiMLorentzVector p4Constr(
        pt * Phase2L1GMT::LSBpt, eta * Phase2L1GMT::LSBeta, phi * Phase2L1GMT::LSBphi, 0.0);
    l1t::SAMuon saMuon(p4Constr, charge, pt, eta, phi, z0, d0, qual);
    saMuon.setTF(mtfType);
    //samuon.setWord(word);

    if (saMuon.hwPt() > 0) {
      saMuons.push_back(saMuon);
    }
  }

  return saMuons;
}

FinalMuons OmtfProcessorPhase2::run(unsigned int iProcessor,
                                    l1t::tftype mtfType,
                                    int bx,
                                    OMTFinputMaker* inputMaker,
                                    std::vector<std::unique_ptr<IOMTFEmulationObserver> >& observers) {
  //uncomment if you want to check execution time of each method
  //boost::timer::auto_cpu_timer t("%ws wall, %us user in getProcessorCandidates\n");

  for (auto& obs : observers)
    obs->observeProcesorBegin(iProcessor, mtfType);

  //input is shared_ptr because the observers may need them after the run() method execution is finished
  std::shared_ptr<OMTFinput> input = std::make_shared<OMTFinput>(omtfConfig);
  inputMaker->buildInputForProcessor(input->getMuonStubs(), iProcessor, mtfType, bx, bx, observers);

  //TODO make a method cleanStubs in OMTFinput
  if (omtfConfig->cleanStubs()) {
    //this has sense for the pattern generation from the tracks with the secondaries
    //if more than one stub is in a given layer, all stubs are removed from this layer
    for (unsigned int iLayer = 0; iLayer < input->getMuonStubs().size(); ++iLayer) {
      auto& layerStubs = input->getMuonStubs()[iLayer];
      int count = std::count_if(layerStubs.begin(), layerStubs.end(), [](auto& ptr) { return ptr != nullptr; });
      if (count > 1) {
        for (auto& ptr : layerStubs)
          ptr.reset();

        LogTrace("OMTFReconstruction") << __FUNCTION__ << ":" << __LINE__ << "cleaning stubs in the layer " << iLayer
                                       << " stubs count :" << count << std::endl;
      }
    }
  }

  //LogTrace("l1tOmtfEventPrint")<<"buildInputForProce "; t.report();
  omtfProc->processInput(iProcessor, mtfType, *(input.get()), observers);

  //LogTrace("l1tOmtfEventPrint")<<"processInput       "; t.report();
  AlgoMuons algoCandidates = omtfProc->sortResults(iProcessor, mtfType);

  //assignQuality must be called after ghostBust, because eta is set there
  /*for (auto& algoMuon : algoCandidates) {
    assignQuality(algoMuon);
  }*/

  if (mlModel) {
    for (auto& algoMuon : algoCandidates) {
      if (algoMuon->isValid()) {
        mlModel->run(algoMuon, observers);
      }
    }
  }

  //LogTrace("l1tOmtfEventPrint")<<"sortResults        "; t.report();
  // perform GB
  //watch out: etaBits2HwEta is used in the ghostBust to convert the AlgoMuons eta, it affect algoCandidates as they are pointers
  AlgoMuons gbCandidates = omtfProc->ghostBust(algoCandidates);

  for (auto& gbCandidate : gbCandidates) {
    assignQualityPhase2(gbCandidate);
  }

  //LogTrace("l1tOmtfEventPrint")<<"ghostBust"; t.report();

  FinalMuons finalMuons = omtfProc->getFinalMuons(iProcessor, mtfType, gbCandidates);

  for (auto& finalMuon : finalMuons) {
    convertToGmtScalesPhase2(iProcessor, mtfType, finalMuon);
  }

  for (auto& finalMuon : finalMuons) {
    finalMuon->setBx(bx);
  }

  for (auto& obs : observers) {
    obs->observeProcesorEmulation(iProcessor, mtfType, input, algoCandidates, gbCandidates, finalMuons);
  }

  return finalMuons;
}
