/*
 * DataROOTDumper.h
 *
 *  Created on: Tue Apr 16 09:57:08 CEST 2019
 *      Author: akalinow
 */

#ifndef OMTF_DATAROOTDUMPER_H_
#define OMTF_DATAROOTDUMPER_H_

#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/IOMTFEmulationObserver.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Tools/PatternOptimizerBase.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"

#include "TMap.h"
#include "TArrayI.h"

#include <functional>

class TTree;

struct OMTFEvent {
public:
  double muonPt, muonEta, muonPhi;
  int muonCharge;

  int omtfCharge, omtfProcessor, omtfScore;
  double omtfPt, omtfEta, omtfPhi;
  unsigned int omtfQuality, omtfRefLayer, omtfHitsWord;

  std::vector<int> hits;
  std::vector<int> hitsQuality;
};

class DataROOTDumper : public PatternOptimizerBase {
public:
  DataROOTDumper(const edm::ParameterSet& edmCfg, const OMTFConfiguration* omtfConfig);

  ~DataROOTDumper() override;

  void observeProcesorEmulation(unsigned int iProcessor,
                                l1t::tftype mtfType,
                                const std::shared_ptr<OMTFinput>& input,
                                const AlgoMuons& algoCandidates,
                                const AlgoMuons& gbCandidates,
                                const std::vector<l1t::RegionalMuonCand>& candMuons) override;

  void observeEventEnd(const edm::Event& iEvent,
                       std::unique_ptr<l1t::RegionalMuonCandBxCollection>& finalCandidates) override;

  void endJob() override;

private:
  void initializeTTree();
  void saveTTree();

  TFile* myFile;
  TTree* myTree;
  OMTFEvent myEvent;
};

#endif /* OMTF_DATAROOTDUMPER_H_ */
