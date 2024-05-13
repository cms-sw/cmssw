/*
 * DataROOTDumper2.h
 *
 *  Created on: Dec 11, 2019
 *      Author: kbunkow
 */

#ifndef L1T_OmtfP1_TOOLS_DATAROOTDUMPER2_H_
#define L1T_OmtfP1_TOOLS_DATAROOTDUMPER2_H_

#include "L1Trigger/L1TMuonOverlapPhase1/interface/Tools/EmulationObserverBase.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Tools/CandidateSimMuonMatcher.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"

#include "TMap.h"
#include "TArrayI.h"
#include "TFile.h"
#include "TH2.h"

#include <functional>

class TTree;

struct OmtfEvent {
public:
  unsigned int eventNum = 0;

  //muonPt = 0 means that no muon was matched to the candidate
  short muonEvent = -1;
  float muonPt = 0, muonEta = 0, muonPhi = 0, muonPropEta = 0, muonPropPhi = 0;
  char muonCharge = 0;
  float muonDxy = 0;
  float muonRho = 0;

  float omtfPt = 0, omtfEta = 0, omtfPhi = 0, omtfUPt = 0;
  char omtfCharge = 0;
  char omtfProcessor = 0;
  short omtfScore = 0;

  short omtfHwEta = 0;

  char omtfQuality = 0;
  char omtfRefLayer = 0;
  char omtfRefHitNum = 0;

  unsigned int omtfFiredLayers = 0;

  bool killed = false;

  float deltaPhi = 0, deltaEta = 0;

  //float omtfPtCont = 0;

  struct Hit {
    union {
      unsigned long rawData = 0;

      struct {
        char layer;
        char quality;
        char z;
        char valid;
        short eta;
        short phiDist;
      };
    };

    ~Hit() {}
  };

  std::vector<unsigned long> hits;
};

class DataROOTDumper2 : public EmulationObserverBase {
public:
  DataROOTDumper2(const edm::ParameterSet& edmCfg,
                  const OMTFConfiguration* omtfConfig,
                  CandidateSimMuonMatcher* candidateSimMuonMatcher);

  ~DataROOTDumper2() override;

  void observeProcesorEmulation(unsigned int iProcessor,
                                l1t::tftype mtfType,
                                const std::shared_ptr<OMTFinput>&,
                                const AlgoMuons& algoCandidates,
                                const AlgoMuons& gbCandidates,
                                const std::vector<l1t::RegionalMuonCand>& candMuons) override;

  void observeEventEnd(const edm::Event& iEvent,
                       std::unique_ptr<l1t::RegionalMuonCandBxCollection>& finalCandidates) override;

  void endJob() override;

private:
  void initializeTTree();

  CandidateSimMuonMatcher* candidateSimMuonMatcher = nullptr;

  TTree* rootTree = nullptr;

  OmtfEvent omtfEvent;

  unsigned int evntCnt = 0;

  TH1I* ptGenPos = nullptr;
  TH1I* ptGenNeg = nullptr;

  std::vector<TH2*> hitVsPt;

  bool dumpKilledOmtfCands = false;

  bool usePropagation = false;
};

#endif /* L1T_OmtfP1_TOOLS_DATAROOTDUMPER2_H_ */
