/*
 * XMLEventWriter.h
 *
 *  Created on: Oct 12, 2017
 *      Author: kbunkow
 */

#ifndef L1T_OmtfP1_XMLEVENTWRITER_H_
#define L1T_OmtfP1_XMLEVENTWRITER_H_

#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/AlgoMuon.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/IOMTFEmulationObserver.h"

#include <boost/property_tree/ptree.hpp>

#include <memory>
#include <string>
#include <vector>

class XMLEventWriter : public IOMTFEmulationObserver {
public:
  XMLEventWriter(const OMTFConfiguration* aOMTFConfig, std::string fName);

  ~XMLEventWriter() override;

  void observeProcesorBegin(unsigned int iProcessor, l1t::tftype mtfType) override;

  void addProcesorData(std::string key, boost::property_tree::ptree& procDataTree) override {
    procTree.add_child(key, procDataTree);
  }

  void observeProcesorEmulation(unsigned int iProcessor,
                                l1t::tftype mtfType,
                                const std::shared_ptr<OMTFinput>& input,
                                const AlgoMuons& algoCandidates,
                                const AlgoMuons& gbCandidates,
                                const FinalMuons& finalMuons) override;

  void observeEventBegin(const edm::Event& iEvent) override;

  void observeEventEnd(const edm::Event& iEvent, FinalMuons& finalMuons) override;

  void endJob() override;

private:
  const OMTFConfiguration* omtfConfig;

  boost::property_tree::ptree tree;

  boost::property_tree::ptree* eventTree = nullptr;

  boost::property_tree::ptree procTree;

  std::string fName;

  unsigned int eventNum = 0;

  unsigned int eventId = 0;
};

#endif /* L1T_OmtfP1_XMLEVENTWRITER_H_ */
