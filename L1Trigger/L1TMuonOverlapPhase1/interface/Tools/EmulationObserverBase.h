/*
 * EmulationObserverBase.h
 *
 *  Created on: Aug 18, 2021
 *      Author: kbunkow
 */

#ifndef L1T_OmtfP1_TOOLS_EMULATIONOBSERVERBASE_H_
#define L1T_OmtfP1_TOOLS_EMULATIONOBSERVERBASE_H_

#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/IOMTFEmulationObserver.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"

class EmulationObserverBase : public IOMTFEmulationObserver  {
public:
  EmulationObserverBase(const edm::ParameterSet& edmCfg, const OMTFConfiguration* omtfConfig);

  virtual ~EmulationObserverBase();

  void observeProcesorEmulation(unsigned int iProcessor,
                                l1t::tftype mtfType,
                                const std::shared_ptr<OMTFinput>& input,
                                const AlgoMuons& algoCandidates,
                                const AlgoMuons& gbCandidates,
                                const std::vector<l1t::RegionalMuonCand>& candMuons) override;

  void observeEventBegin(const edm::Event& iEvent) override;

  //void observeEventEnd(const edm::Event& iEvent,
  //                     std::unique_ptr<l1t::RegionalMuonCandBxCollection>& finalCandidates) override;

  //void endJob() override;

  const SimTrack* findSimMuon(const edm::Event& event, const SimTrack* previous = nullptr);

protected:
  edm::ParameterSet edmCfg;
  const OMTFConfiguration* omtfConfig;

  const SimTrack* simMuon = nullptr;

  //candidate found by omtf in a given event
  AlgoMuons::value_type omtfCand;

  l1t::RegionalMuonCand regionalMuonCand;

  //AlgoMuons algoCandidates;

  unsigned int candProcIndx = 0;
};

#endif /* INTERFACE_TOOLS_EMULATIONOBSERVERBASE_H_ */
