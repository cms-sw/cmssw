/*
 * IOMTFReconstructionObserver.h
 *
 *  Created on: Oct 12, 2017
 *      Author: kbunkow
 */

#ifndef OMTF_IOMTFRECONSTRUCTIONOBSERVER_H_
#define OMTF_IOMTFRECONSTRUCTIONOBSERVER_H_

#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/AlgoMuon.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OMTFinput.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"
#include <memory>
#include <vector>

namespace edm {
  class Event;
} /* namespace edm */

class IOMTFEmulationObserver {
public:
  IOMTFEmulationObserver();
  virtual ~IOMTFEmulationObserver();

  virtual void beginRun(edm::EventSetup const& eventSetup) {}

  virtual void observeProcesorEmulation(unsigned int iProcessor,
                                        l1t::tftype mtfType,
                                        const std::shared_ptr<OMTFinput>& input,
                                        const AlgoMuons& algoCandidates,
                                        const AlgoMuons& gbCandidates,
                                        const std::vector<l1t::RegionalMuonCand>& candMuons) = 0;

  virtual void observeEventBegin(const edm::Event& iEvent){};

  virtual void observeEventEnd(const edm::Event& iEvent,
                               std::unique_ptr<l1t::RegionalMuonCandBxCollection>& finalCandidates){};

  virtual void endJob() = 0;
};

#endif /* OMTF_IOMTFRECONSTRUCTIONOBSERVER_H_ */
