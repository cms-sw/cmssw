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

#include "L1Trigger/DTPhase2Trigger/interface/muonpath.h"
#include "L1Trigger/DTPhase2Trigger/interface/analtypedefs.h"
#include "L1Trigger/DTPhase2Trigger/interface/constants.h"

#include "L1Trigger/DTPhase2Trigger/interface/MotherGrouping.h"

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

#include "L1Trigger/DTPhase2Trigger/interface/Pattern.h"
#include "L1Trigger/DTPhase2Trigger/interface/CandidateGroup.h"
#include "TFile.h"
#include "TString.h"

#include "L1Trigger/DTPhase2Trigger/interface/analtypedefs.h"

// ===============================================================================
// Class declarations
// ===============================================================================

class PseudoBayesGrouping : public MotherGrouping {
  public:
    // Constructors and destructor
    PseudoBayesGrouping(const edm::ParameterSet& pset);
    ~PseudoBayesGrouping() override;
    
    // Main methods
    void initialise(const edm::EventSetup& iEventSetup) override;
    void run(edm::Event& iEvent, const edm::EventSetup& iEventSetup, DTDigiCollection digis, std::vector<MuonPath*> *outMpath) override;
    void finish() override;
    
    // Other public methods
    
    // Public attributes
    
  private:
    // Private methods
    void LoadPattern(std::vector<std::vector<std::vector<int> > >::iterator itPattern);
    void FillDigisByLayer(DTDigiCollection *digis);
    void CleanDigisByLayer();
    void RecognisePatternsByLayerPairs();
    void RecognisePatterns(std::vector<DTPrimitive> digisinLDown, std::vector<DTPrimitive> digisinLUp, std::vector<Pattern*> patterns);
    void ReCleanPatternsAndDigis();
    void FillMuonPaths( std::vector<MuonPath*> *mpaths);

    //Comparator for pointer mode
    struct CandPointGreat {
      bool operator()(CandidateGroup* c1, CandidateGroup* c2) {
        return *c1 > *c2;
      }
    };

    // Private attributes
    // Config variables
    Bool_t debug;
    std::string pattern_filename;
    int  pidx;
    int  minNLayerHits;
    int  allowedVariance;
    bool allowDuplicates;
    bool allowUncorrelatedPatterns;
    bool setLateralities;
    bool saveOnPlace;
    int  minSingleSLHitsMax;
    int  minSingleSLHitsMin;
    int  minUncorrelatedHits;

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
    std::vector<CandidateGroup*>* prelimMatches;
    std::vector<CandidateGroup*>* allMatches;
    std::vector<CandidateGroup*>* finalMatches;

    //Pattern related info
    int nPatterns;
    std::vector<Pattern*> allPatterns;

    std::vector<Pattern*> L0L7Patterns;
    std::vector<Pattern*> L1L7Patterns;
    std::vector<Pattern*> L2L7Patterns;
    std::vector<Pattern*> L3L7Patterns;
    std::vector<Pattern*> L4L7Patterns;
    std::vector<Pattern*> L5L7Patterns;
    std::vector<Pattern*> L6L7Patterns;

    std::vector<Pattern*> L0L6Patterns;
    std::vector<Pattern*> L1L6Patterns;
    std::vector<Pattern*> L2L6Patterns;
    std::vector<Pattern*> L3L6Patterns;
    std::vector<Pattern*> L4L6Patterns;
    std::vector<Pattern*> L5L6Patterns;

    std::vector<Pattern*> L0L5Patterns;
    std::vector<Pattern*> L1L5Patterns;
    std::vector<Pattern*> L2L5Patterns;
    std::vector<Pattern*> L3L5Patterns;
    std::vector<Pattern*> L4L5Patterns;

    std::vector<Pattern*> L0L4Patterns;
    std::vector<Pattern*> L1L4Patterns;
    std::vector<Pattern*> L2L4Patterns;
    std::vector<Pattern*> L3L4Patterns;

    std::vector<Pattern*> L0L3Patterns;
    std::vector<Pattern*> L1L3Patterns;
    std::vector<Pattern*> L2L3Patterns;

    std::vector<Pattern*> L0L2Patterns;
    std::vector<Pattern*> L1L2Patterns;

    std::vector<Pattern*> L0L1Patterns;

    CandidateGroup *cand;
};


#endif
