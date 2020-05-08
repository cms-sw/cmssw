#ifndef Phase2L1Trigger_DTTrigger_PseudoBayesGrouping_cc
#define Phase2L1Trigger_DTTrigger_PseudoBayesGrouping_cc

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThDigi.h"

#include "L1Trigger/DTTriggerPhase2/interface/MuonPath.h"
#include "L1Trigger/DTTriggerPhase2/interface/constants.h"

#include "L1Trigger/DTTriggerPhase2/interface/MotherGrouping.h"

// #include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambContainer.h"
// #include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambDigi.h"

#include "CalibMuon/DTDigiSync/interface/DTTTrigBaseSync.h"
#include "CalibMuon/DTDigiSync/interface/DTTTrigSyncFactory.h"

#include "L1Trigger/DTSectorCollector/interface/DTSectCollPhSegm.h"
#include "L1Trigger/DTSectorCollector/interface/DTSectCollThSegm.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"

#include <iostream>
#include <fstream>

#include "L1Trigger/DTTriggerPhase2/interface/DTPattern.h"
#include "L1Trigger/DTTriggerPhase2/interface/CandidateGroup.h"
#include "TFile.h"
#include "TString.h"

// ===============================================================================
// Class declarations
// ===============================================================================

class PseudoBayesGrouping : public MotherGrouping {
public:
  // Constructors and destructor
  PseudoBayesGrouping(const edm::ParameterSet& pset, edm::ConsumesCollector& iC);
  ~PseudoBayesGrouping() override;

  // Main methods
  void initialise(const edm::EventSetup& iEventSetup) override;
  void run(edm::Event& iEvent,
           const edm::EventSetup& iEventSetup,
           const DTDigiCollection& digis,
           std::vector<MuonPath*>* outMpath) override;
  void finish() override;

  // Other public methods

  // Public attributes

private:
  // Private methods
  void LoadPattern(std::vector<std::vector<std::vector<int>>>::iterator itPattern);
  void FillDigisByLayer(const DTDigiCollection* digis);
  void CleanDigisByLayer();
  void RecognisePatternsByLayerPairs();
  void RecognisePatterns(std::vector<DTPrimitive> digisinLDown,
                         std::vector<DTPrimitive> digisinLUp,
                         std::vector<DTPattern*> patterns);
  void ReCleanPatternsAndDigis();
  void FillMuonPaths(std::vector<MuonPath*>* mpaths);

  //Comparator for pointer mode
  struct CandPointGreat {
    bool operator()(CandidateGroup* c1, CandidateGroup* c2) { return *c1 > *c2; }
  };

  // Private attributes
  // Config variables
  bool debug;
  std::string pattern_filename;
  int pidx;
  int minNLayerHits;
  int allowedVariance;
  bool allowDuplicates;
  bool allowUncorrelatedPatterns;
  bool setLateralities;
  bool saveOnPlace;
  int minSingleSLHitsMax;
  int minSingleSLHitsMin;
  int minUncorrelatedHits;

  //Classified digis
  std::vector<DTPrimitive> alldigis;

  std::vector<DTPrimitive> digisinL0;
  std::vector<DTPrimitive> digisinL1;
  std::vector<DTPrimitive> digisinL2;
  std::vector<DTPrimitive> digisinL3;
  std::vector<DTPrimitive> digisinL4;
  std::vector<DTPrimitive> digisinL5;
  std::vector<DTPrimitive> digisinL6;
  std::vector<DTPrimitive> digisinL7;

  //Preliminary matches, those can grow quite big so better not to rely on the stack
  std::unique_ptr<std::vector<CandidateGroup*>> prelimMatches;
  std::unique_ptr<std::vector<CandidateGroup*>> allMatches;
  std::unique_ptr<std::vector<CandidateGroup*>> finalMatches;

  //Pattern related info
  int nPatterns;
  std::vector<DTPattern*> allPatterns;

  std::vector<DTPattern*> L0L7Patterns;
  std::vector<DTPattern*> L1L7Patterns;
  std::vector<DTPattern*> L2L7Patterns;
  std::vector<DTPattern*> L3L7Patterns;
  std::vector<DTPattern*> L4L7Patterns;
  std::vector<DTPattern*> L5L7Patterns;
  std::vector<DTPattern*> L6L7Patterns;

  std::vector<DTPattern*> L0L6Patterns;
  std::vector<DTPattern*> L1L6Patterns;
  std::vector<DTPattern*> L2L6Patterns;
  std::vector<DTPattern*> L3L6Patterns;
  std::vector<DTPattern*> L4L6Patterns;
  std::vector<DTPattern*> L5L6Patterns;

  std::vector<DTPattern*> L0L5Patterns;
  std::vector<DTPattern*> L1L5Patterns;
  std::vector<DTPattern*> L2L5Patterns;
  std::vector<DTPattern*> L3L5Patterns;
  std::vector<DTPattern*> L4L5Patterns;

  std::vector<DTPattern*> L0L4Patterns;
  std::vector<DTPattern*> L1L4Patterns;
  std::vector<DTPattern*> L2L4Patterns;
  std::vector<DTPattern*> L3L4Patterns;

  std::vector<DTPattern*> L0L3Patterns;
  std::vector<DTPattern*> L1L3Patterns;
  std::vector<DTPattern*> L2L3Patterns;

  std::vector<DTPattern*> L0L2Patterns;
  std::vector<DTPattern*> L1L2Patterns;

  std::vector<DTPattern*> L0L1Patterns;

  CandidateGroup* cand;
};

#endif
