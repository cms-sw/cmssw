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

XMLEventWriter::XMLEventWriter(const OMTFConfiguration* aOMTFConfig, std::string fName)
    : omtfConfig(aOMTFConfig), xmlWriter(aOMTFConfig), currentElement(nullptr), fName(fName) {
  //std::string fName = "OMTF";
  xmlWriter.initialiseXMLDocument("OMTF");
  eventNum = 0;
};

XMLEventWriter::~XMLEventWriter() {}

void XMLEventWriter::observeProcesorEmulation(unsigned int iProcessor,
                                              l1t::tftype mtfType,
                                              const std::shared_ptr<OMTFinput>& input,
                                              const AlgoMuons& algoCandidates,
                                              const AlgoMuons& gbCandidates,
                                              const std::vector<l1t::RegionalMuonCand>& candMuons) {
  if (eventNum > 5000)
    return;

  int endcap = (mtfType == l1t::omtf_neg) ? -1 : ((mtfType == l1t::omtf_pos) ? +1 : 0);
  OmtfName board(iProcessor, endcap);

  if (candMuons.empty())
    return;

  //if(currentElement == nullptr)
  //  currentElement = xmlWriter.writeEventHeader(eventId);

  xercesc::DOMElement* aProcElement = xmlWriter.writeEventData(currentElement, board, *(input.get()));

  for (auto& algoCand : algoCandidates) {
    ///Dump only regions, where a candidate was found
    if (algoCand->isValid()) {
      xmlWriter.writeAlgoMuon(aProcElement, *algoCand);
      /*if(dumpDetailedResultToXML){
        for(auto & itKey: results[iRefHit])
          xmlWriter.writeResultsData(aProcElement, iRefHit, itKey.first,itKey.second);
      }*/
    }
  }

  for (auto& candMuon : candMuons)
    xmlWriter.writeCandMuon(aProcElement, candMuon);
}

void XMLEventWriter::observeEventBegin(const edm::Event& iEvent) {
  eventNum++;
  if (eventNum > 5000)
    //due to some bug if more events is written the memory consumption s very big and program crashes
    return;
  //currentElement = xmlWriter.writeEventHeader(iEvent.id().event());
  eventId = iEvent.id().event();
  currentElement = xmlWriter.writeEventHeader(eventId);
}

void XMLEventWriter::observeEventEnd(const edm::Event& iEvent,
                                     std::unique_ptr<l1t::RegionalMuonCandBxCollection>& finalCandidates) {
  currentElement = nullptr;
}

void XMLEventWriter::endJob() { xmlWriter.finaliseXMLDocument(fName); }
