/*
 * DataROOTDumper2.h
 *
 *  Created on: Dec 11, 2019
 *      Author: kbunkow
 */

#ifndef L1T_OmtfP1_TOOLS_DATAROOTDUMPER2_H_
#define L1T_OmtfP1_TOOLS_DATAROOTDUMPER2_H_

#include "L1Trigger/L1TMuonOverlapPhase1/interface/Tools/EmulationObserverBase.h"

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
  double muonPt = 0, muonEta = 0, muonPhi = 0;
  int muonCharge = 0;

  int omtfCharge = 0, omtfProcessor = 0, omtfScore = 0;
  double omtfPt = 0, omtfEta = 0, omtfPhi = 0;
  unsigned int omtfQuality = 0, omtfRefLayer = 0;
  unsigned int omtfFiredLayers = 0;

  float omtfPtCont = 0;

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

  //OmtfEvent() {}
};

class DataROOTDumper2 : public EmulationObserverBase {
public:
  DataROOTDumper2(const edm::ParameterSet& edmCfg, const OMTFConfiguration* omtfConfig, std::string rootFileName);

  ~DataROOTDumper2() override;

  void observeEventEnd(const edm::Event& iEvent,
                       std::unique_ptr<l1t::RegionalMuonCandBxCollection>& finalCandidates) override;

  void endJob() override;

private:
  void initializeTTree(std::string rootFileName);
  void saveTTree();

  TFile* rootFile = nullptr;
  TTree* rootTree = nullptr;

  OmtfEvent omtfEvent;

  unsigned int evntCnt = 0;

  TH1I* ptGenPos = nullptr;
  TH1I* ptGenNeg = nullptr;

  std::vector<TH2*> hitVsPt;
};

#endif /* L1T_OmtfP1_TOOLS_DATAROOTDUMPER2_H_ */
