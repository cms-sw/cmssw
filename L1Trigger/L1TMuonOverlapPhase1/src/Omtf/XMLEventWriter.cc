/*
 * XMLEventWriter.cc
 *
 *  Created on: Oct 12, 2017
 *      Author: kbunkow
 */

#include "DataFormats/Provenance/interface/EventID.h"
#include "FWCore/Common/interface/EventBase.h"
#include "FWCore/Framework/interface/Event.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OmtfName.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OMTFinput.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/XMLEventWriter.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <boost/property_tree/xml_parser.hpp>

#include <bitset>

XMLEventWriter::XMLEventWriter(const OMTFConfiguration* aOMTFConfig, std::string fName)
    : omtfConfig(aOMTFConfig), fName(fName) {
  //std::string fName = "OMTF";
  eventNum = 0;

  unsigned int version = aOMTFConfig->patternsVersion();
  const unsigned int mask16bits = 0xFFFF;

  version &= mask16bits;

  std::ostringstream stringStr;
  stringStr.str("");
  stringStr << "0x" << std::hex << std::setfill('0') << std::setw(4) << version;

  tree.put("OMTF.<xmlattr>.version", stringStr.str());
};

XMLEventWriter::~XMLEventWriter() {}

void XMLEventWriter::observeProcesorBegin(unsigned int iProcessor, l1t::tftype mtfType) {
  procTree.clear();

  int endcap = (mtfType == l1t::omtf_neg) ? -1 : ((mtfType == l1t::omtf_pos) ? +1 : 0);
  OmtfName board(iProcessor, endcap, omtfConfig);
  procTree.add("<xmlattr>.board", board.name());
  procTree.add("<xmlattr>.iProcessor", iProcessor);

  std::ostringstream stringStr;
  stringStr << (board.position() == 1 ? "+" : "") << board.position();
  procTree.add("<xmlattr>.position", stringStr.str());
}

void XMLEventWriter::observeProcesorEmulation(unsigned int iProcessor,
                                              l1t::tftype mtfType,
                                              const std::shared_ptr<OMTFinput>& input,
                                              const AlgoMuons& algoCandidates,
                                              const AlgoMuons& gbCandidates,
                                              const FinalMuons& finalMuons) {
  int endcap = (mtfType == l1t::omtf_neg) ? -1 : ((mtfType == l1t::omtf_pos) ? +1 : 0);
  OmtfName board(iProcessor, endcap, omtfConfig);

  if (finalMuons.empty())
    return;

  for (unsigned int iLayer = 0; iLayer < omtfConfig->nLayers(); ++iLayer) {
    boost::property_tree::ptree layerTree;

    for (unsigned int iHit = 0; iHit < input->getMuonStubs()[iLayer].size(); ++iHit) {
      int hitPhi = input->getPhiHw(iLayer, iHit);
      if (hitPhi >= (int)omtfConfig->nPhiBins())
        continue;

      auto& hitTree = layerTree.add("Hit", "");

      hitTree.add("<xmlattr>.iInput", iHit);
      hitTree.add("<xmlattr>.iEta", input->getHitEta(iLayer, iHit));
      hitTree.add("<xmlattr>.iPhi", hitPhi);

      //in the firmware the hit quality is taken from link data only for the DT stubs
      //for the CSC and RPC 1 means the hit is valid, 0 - not.
      //in the input it is still worth to have the actual quality of the CSC and RPC
      //Because it might be used in the neural network
      if (iLayer >= 6)
        hitTree.add("<xmlattr>.iQual", 1);
      else
        hitTree.add("<xmlattr>.iQual", input->getHitQual(iLayer, iHit));
    }

    if (!layerTree.empty()) {
      layerTree.add("<xmlattr>.iLayer", iLayer);
      procTree.add_child("Layer", layerTree);
    }
  }

  for (auto& algoCand : algoCandidates) {
    ///Dump only regions, where a candidate was found
    if (algoCand->isValid()) {
      auto& algoMuonTree = procTree.add("AlgoMuon", "");
      algoMuonTree.add("<xmlattr>.charge", algoCand->getChargeConstr());
      algoMuonTree.add("<xmlattr>.disc", algoCand->getDisc());
      algoMuonTree.add("<xmlattr>.pdfSumConstr", algoCand->getGpResultConstr().getPdfSum());
      algoMuonTree.add("<xmlattr>.pdfSumUnconstr", algoCand->getGpResultUnconstr().getPdfSumUnconstr());
      algoMuonTree.add("<xmlattr>.etaCode", algoCand->getEtaHw());
      algoMuonTree.add("<xmlattr>.iRefHit", algoCand->getRefHitNumber());
      algoMuonTree.add("<xmlattr>.iRefLayer", algoCand->getRefLayer());

      //algoMuonTree.add("<xmlattr>.layers", std::bitset<18>(algoCand->getFiredLayerBits()));

      algoMuonTree.add("<xmlattr>.layersConstr", std::bitset<18>(algoCand->getGpResultConstr().getFiredLayerBits()));
      algoMuonTree.add("<xmlattr>.layersUnconstr",
                       std::bitset<18>(algoCand->getGpResultUnconstr().getFiredLayerBits()));

      algoMuonTree.add("<xmlattr>.nHits", algoCand->getQ());

      algoMuonTree.add("<xmlattr>.firedCntConstr", algoCand->getGpResultConstr().getFiredLayerCnt());
      algoMuonTree.add("<xmlattr>.firedCntUnconstr", algoCand->getGpResultUnconstr().getFiredLayerCnt());

      algoMuonTree.add("<xmlattr>.patNumConstr", algoCand->getHwPatternNumConstr());
      algoMuonTree.add("<xmlattr>.patNumUnconstr", algoCand->getHwPatternNumUnconstr());

      algoMuonTree.add("<xmlattr>.phiCode", algoCand->getPhi());

      algoMuonTree.add("<xmlattr>.phiConstr", algoCand->getGpResultConstr().getPhi());
      algoMuonTree.add("<xmlattr>.phiUnConstr", algoCand->getGpResultUnconstr().getPhi());

      algoMuonTree.add("<xmlattr>.phiRHit", algoCand->getPhiRHit());

      //in the firmware, the algoMuon has no pt nor upt yet,
      //only the pattern number, which is converted to the hwpt in the ghostbuster
      algoMuonTree.add("<xmlattr>.ptCodeConstr", algoCand->getPtConstr());
      algoMuonTree.add("<xmlattr>.ptCodeUnconstr", algoCand->getPtUnconstr());

      auto& gpResultTree = algoMuonTree.add("gpResultConstr", "");
      auto& gpResultConstr = algoCand->getGpResultConstr();

      gpResultTree.add("<xmlattr>.patNum", algoCand->getHwPatternNumConstr());
      gpResultTree.add("<xmlattr>.pdfSum", gpResultConstr.getPdfSum());

      for (unsigned int iLogicLayer = 0; iLogicLayer < gpResultConstr.getStubResults().size(); ++iLogicLayer) {
        auto& layerTree = gpResultTree.add("layer", "");
        layerTree.add("<xmlattr>.num", iLogicLayer);
        auto pdfBin = gpResultConstr.getStubResults()[iLogicLayer].getPdfBin();
        if (pdfBin == 5400)
          pdfBin = 0;
        layerTree.add("<xmlattr>.pdfBin", pdfBin);
        layerTree.add("<xmlattr>.pdfVal", gpResultConstr.getStubResults()[iLogicLayer].getPdfVal());
        layerTree.add("<xmlattr>.fired", gpResultConstr.isLayerFired(iLogicLayer));
      }

      if (algoCand->getGpResultUnconstr().isValid()) {
        auto& gpResultTree = algoMuonTree.add("gpResultUnconstr", "");
        auto& gpResult = algoCand->getGpResultUnconstr();

        gpResultTree.add("<xmlattr>.patNum", algoCand->getHwPatternNumUnconstr());
        gpResultTree.add("<xmlattr>.pdfSum", gpResult.getPdfSumUnconstr());

        for (unsigned int iLogicLayer = 0; iLogicLayer < gpResult.getStubResults().size(); ++iLogicLayer) {
          auto& layerTree = gpResultTree.add("layer", "");
          layerTree.add("<xmlattr>.num", iLogicLayer);
          auto pdfBin = gpResult.getStubResults()[iLogicLayer].getPdfBin();
          if (pdfBin == 5400)
            pdfBin = 0;
          layerTree.add("<xmlattr>.pdfBin", pdfBin);
          layerTree.add("<xmlattr>.pdfVal", gpResult.getStubResults()[iLogicLayer].getPdfVal());
          layerTree.add("<xmlattr>.fired", gpResult.isLayerFired(iLogicLayer));
        }
      }
    }
  }

  for (auto& finalMuon : finalMuons) {
    auto& candMuonTree = procTree.add("CandMuon", "");
    candMuonTree.add("<xmlattr>.hwEta", finalMuon->getEtaGmt());
    candMuonTree.add("<xmlattr>.hwPhi", finalMuon->getPhiGmt());
    candMuonTree.add("<xmlattr>.hwPt", finalMuon->getPtGmt());
    candMuonTree.add("<xmlattr>.hwUPt", finalMuon->getPtUnconstrGmt());
    candMuonTree.add("<xmlattr>.hwQual", finalMuon->getQuality());
    candMuonTree.add("<xmlattr>.hwSign", finalMuon->getSign());
    candMuonTree.add("<xmlattr>.hwSignValid", 1);
    candMuonTree.add("<xmlattr>.hwTrackAddress", std::bitset<29>(finalMuon->getAlgoMuon()->getFiredLayerBits()));
    candMuonTree.add("<xmlattr>.link", (mtfType == l1t::omtf_neg ? 60 + iProcessor : 42 + iProcessor));
    candMuonTree.add("<xmlattr>.processor", iProcessor);

    std::ostringstream stringStr;
    if (mtfType == l1t::omtf_neg)
      stringStr << "OMTF_NEG";
    else if (mtfType == l1t::omtf_pos)
      stringStr << "OMTF_POS";

    candMuonTree.add("<xmlattr>.trackFinderType", stringStr.str());
  }

  if (!procTree.empty())
    eventTree->add_child("Processor", procTree);
}

void XMLEventWriter::observeEventBegin(const edm::Event& iEvent) {
  eventNum++;
  eventId = iEvent.id().event();

  eventTree = &(tree.add("OMTF.Event", ""));
  eventTree->add("<xmlattr>.iEvent", eventId);

  eventTree = &(eventTree->add("bx", ""));
  eventTree->add("<xmlattr>.iBx", 2 * eventId);
}

void XMLEventWriter::observeEventEnd(const edm::Event& iEvent, FinalMuons& finalMuons) {}

void XMLEventWriter::endJob() {
  edm::LogInfo("l1tOmtfEventPrint") << "XMLEventWriter::endJob() - writing the data to the xml - starting";
  boost::property_tree::write_xml(
      fName, tree, std::locale(), boost::property_tree::xml_parser::xml_writer_make_settings<std::string>(' ', 2));
  edm::LogInfo("l1tOmtfEventPrint") << "XMLEventWriter::endJob() - writing the data to the xml - done";
}
