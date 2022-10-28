#ifndef Phase2L1Trigger_DTTrigger_PseudoBayesGrouping_h
#define Phase2L1Trigger_DTTrigger_PseudoBayesGrouping_h

#include "L1Trigger/DTTriggerPhase2/interface/MotherGrouping.h"
#include "L1Trigger/DTTriggerPhase2/interface/DTPattern.h"
#include "L1Trigger/DTTriggerPhase2/interface/CandidateGroup.h"

// ===============================================================================
// Class declarations
// ===============================================================================
// using namespace dtbayesam;

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
           MuonPathPtrs& outMpath) override;
  void finish() override;

  // Other public methods

  // Public attributes

private:
  // Private methods
  void LoadPattern(TString pattern_file_name, int MB_number_input, int SL_shift);
  void FillDigisByLayer(const DTDigiCollection* digis);
  void CleanDigisByLayer();
  void RecognisePatternsByLayerPairs(DTChamberId chamber_ID);
  void RecognisePatterns(std::vector<DTPrimitive> digisinLDown,
                         std::vector<DTPrimitive> digisinLUp,
                         dtbayesam::DTPatternPtrs patterns);
  void ReCleanPatternsAndDigis();
  void FillMuonPaths(MuonPathPtrs& mpaths);

  //Comparator for pointer mode
  struct CandPointGreat {
    bool operator()(dtbayesam::CandidateGroupPtr c1, dtbayesam::CandidateGroupPtr c2) { return *c1 > *c2; }
  };

  // Private attributes
  // Config variables
  const bool debug_;
  std::string pattern_filename_;
  int pidx_;
  int minNLayerHits_;
  int allowedVariance_;
  bool allowDuplicates_;
  bool allowUncorrelatedPatterns_;
  bool setLateralities_;
  bool saveOnPlace_;
  int minSingleSLHitsMax_;
  int minSingleSLHitsMin_;
  int minUncorrelatedHits_;
  uint maxPathsPerMatch_;

  //Classified digis
  std::vector<DTPrimitive> alldigis_;

  std::vector<DTPrimitive> digisinL0_;
  std::vector<DTPrimitive> digisinL1_;
  std::vector<DTPrimitive> digisinL2_;
  std::vector<DTPrimitive> digisinL3_;
  std::vector<DTPrimitive> digisinL4_;
  std::vector<DTPrimitive> digisinL5_;
  std::vector<DTPrimitive> digisinL6_;
  std::vector<DTPrimitive> digisinL7_;

  //Preliminary matches, those can grow quite big so better not to rely on the stack
  std::unique_ptr<dtbayesam::CandidateGroupPtrs> prelimMatches_;
  std::unique_ptr<dtbayesam::CandidateGroupPtrs> allMatches_;
  std::unique_ptr<dtbayesam::CandidateGroupPtrs> finalMatches_;

  //Pattern related info
  int nPatterns_;

  // One vector per MB and per shift of SL3 wrt SL1
  // shift = (0, 1, 2) --> (left, nothing, right)
  dtbayesam::DTPatternPtrs allPatterns_[4][3];

  dtbayesam::DTPatternPtrs L0L7Patterns_[4][3];
  dtbayesam::DTPatternPtrs L1L7Patterns_[4][3];
  dtbayesam::DTPatternPtrs L2L7Patterns_[4][3];
  dtbayesam::DTPatternPtrs L3L7Patterns_[4][3];
  dtbayesam::DTPatternPtrs L4L7Patterns_[4][3];
  dtbayesam::DTPatternPtrs L5L7Patterns_[4][3];
  dtbayesam::DTPatternPtrs L6L7Patterns_[4][3];

  dtbayesam::DTPatternPtrs L0L6Patterns_[4][3];
  dtbayesam::DTPatternPtrs L1L6Patterns_[4][3];
  dtbayesam::DTPatternPtrs L2L6Patterns_[4][3];
  dtbayesam::DTPatternPtrs L3L6Patterns_[4][3];
  dtbayesam::DTPatternPtrs L4L6Patterns_[4][3];
  dtbayesam::DTPatternPtrs L5L6Patterns_[4][3];

  dtbayesam::DTPatternPtrs L0L5Patterns_[4][3];
  dtbayesam::DTPatternPtrs L1L5Patterns_[4][3];
  dtbayesam::DTPatternPtrs L2L5Patterns_[4][3];
  dtbayesam::DTPatternPtrs L3L5Patterns_[4][3];
  dtbayesam::DTPatternPtrs L4L5Patterns_[4][3];

  dtbayesam::DTPatternPtrs L0L4Patterns_[4][3];
  dtbayesam::DTPatternPtrs L1L4Patterns_[4][3];
  dtbayesam::DTPatternPtrs L2L4Patterns_[4][3];
  dtbayesam::DTPatternPtrs L3L4Patterns_[4][3];

  dtbayesam::DTPatternPtrs L0L3Patterns_[4][3];
  dtbayesam::DTPatternPtrs L1L3Patterns_[4][3];
  dtbayesam::DTPatternPtrs L2L3Patterns_[4][3];

  dtbayesam::DTPatternPtrs L0L2Patterns_[4][3];
  dtbayesam::DTPatternPtrs L1L2Patterns_[4][3];

  dtbayesam::DTPatternPtrs L0L1Patterns_[4][3];
};

#endif
