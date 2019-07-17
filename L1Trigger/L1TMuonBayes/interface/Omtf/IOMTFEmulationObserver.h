/*
 * IOMTFReconstructionObserver.h
 *
 *  Created on: Oct 12, 2017
 *      Author: kbunkow
 */

#ifndef OMTF_IOMTFRECONSTRUCTIONOBSERVER_H_
#define OMTF_IOMTFRECONSTRUCTIONOBSERVER_H_

#include <L1Trigger/L1TMuonBayes/interface/Omtf/AlgoMuon.h>
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"

class IOMTFEmulationObserver {
public:
  IOMTFEmulationObserver();
  virtual ~IOMTFEmulationObserver();

  virtual void observeProcesorEmulation(unsigned int iProcessor, l1t::tftype mtfType,  const OMTFinput &input,
      const AlgoMuons& algoCandidates,
      const AlgoMuons& gbCandidates,
      const std::vector<l1t::RegionalMuonCand> & candMuons ) = 0;

  virtual void observeEventBegin(const edm::Event& iEvent) {};

  virtual void observeEventEnd(const edm::Event& iEvent) {};

  virtual void endJob() = 0;
};

#endif /* OMTF_IOMTFRECONSTRUCTIONOBSERVER_H_ */
