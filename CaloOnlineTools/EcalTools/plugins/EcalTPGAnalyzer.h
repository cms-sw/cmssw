#ifndef CaloOnlineTools_EcalTools_EcalTPGAnalyzer_h
#define CaloOnlineTools_EcalTools_EcalTPGAnalyzer_h
// -*- C++ -*-
//
// Class:      EcalTPGAnalyzer
//
//
// Original Author:  Pascal Paganini
//
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMask.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskAlgoTrigRcd.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDigi/interface/EcalTriggerPrimitiveDigi.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include <vector>
#include <string>
#include <TFile.h>
#include <TTree.h>

class CaloSubdetectorGeometry;

// Auxiliary class
class towerEner {
public:
  float eRec_;
  int tpgEmul_[5];
  int tpgADC_;
  int iphi_, ieta_, nbXtal_;
  towerEner() : eRec_(0), tpgADC_(0), iphi_(-999), ieta_(-999), nbXtal_(0) {
    for (int i = 0; i < 5; i++)
      tpgEmul_[i] = 0;
  }
};

class EcalTPGAnalyzer : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit EcalTPGAnalyzer(const edm::ParameterSet &);
  ~EcalTPGAnalyzer() override;
  void analyze(edm::Event const &, edm::EventSetup const &) override;
  void beginRun(edm::Run const &, edm::EventSetup const &) override;
  void endRun(edm::Run const &, edm::EventSetup const &) override;

private:
  struct EcalTPGVariables {
    // event variables
    unsigned int runNb;
    unsigned int evtNb;
    unsigned int bxNb;
    unsigned int orbitNb;
    unsigned int nbOfActiveTriggers;
    int activeTriggers[128];

    // tower variables
    unsigned int nbOfTowers;  //max 4032 EB+EE
    int ieta[4032];
    int iphi[4032];
    int nbOfXtals[4032];
    int rawTPData[4032];
    int rawTPEmul1[4032];
    int rawTPEmul2[4032];
    int rawTPEmul3[4032];
    int rawTPEmul4[4032];
    int rawTPEmul5[4032];
    float eRec[4032];
  };

private:
  TFile *file_;
  TTree *tree_;
  EcalTPGVariables treeVariables_;

  const edm::InputTag tpCollection_;
  const edm::InputTag tpEmulatorCollection_;
  const edm::InputTag digiCollectionEB_;
  const edm::InputTag digiCollectionEE_;
  const std::string gtRecordCollectionTag_;

  const edm::EDGetTokenT<L1GlobalTriggerReadoutRecord> l1GtReadoutRecordToken_;
  const edm::EDGetTokenT<EcalTrigPrimDigiCollection> tpToken_;
  const edm::EDGetTokenT<EcalTrigPrimDigiCollection> tpEmulToken_;
  const edm::EDGetTokenT<EBDigiCollection> ebDigiToken_;
  const edm::EDGetTokenT<EEDigiCollection> eeDigiToken_;

  const edm::ESGetToken<EcalTrigTowerConstituentsMap, IdealGeometryRecord> eTTMapToken_;
  const edm::ESGetToken<CaloSubdetectorGeometry, EcalBarrelGeometryRecord> ebGeometryToken_;
  const edm::ESGetToken<CaloSubdetectorGeometry, EcalEndcapGeometryRecord> eeGeometryToken_;
  const edm::ESGetToken<L1GtTriggerMask, L1GtTriggerMaskAlgoTrigRcd> l1GtMaskToken_;

  const bool allowTP_;
  const bool useEE_;
  const bool print_;

  const CaloSubdetectorGeometry *theBarrelGeometry_;
  const CaloSubdetectorGeometry *theEndcapGeometry_;
  const EcalTrigTowerConstituentsMap *eTTmap_;
};

#endif
