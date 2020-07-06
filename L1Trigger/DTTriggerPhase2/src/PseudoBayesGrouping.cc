#include <memory>

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TFile.h"

#include "L1Trigger/DTTriggerPhase2/interface/PseudoBayesGrouping.h"

using namespace edm;
using namespace std;
using namespace cmsdt;
using namespace dtbayesam;
// ============================================================================
// Constructors and destructor
// ============================================================================
PseudoBayesGrouping::PseudoBayesGrouping(const ParameterSet& pset, edm::ConsumesCollector& iC)
    : MotherGrouping(pset, iC) {
  // Obtention of parameters
  debug_ = pset.getUntrackedParameter<bool>("debug");
  pattern_filename_ = pset.getUntrackedParameter<edm::FileInPath>("pattern_filename").fullPath();
  minNLayerHits_ = pset.getUntrackedParameter<int>("minNLayerHits");
  minSingleSLHitsMax_ = pset.getUntrackedParameter<int>("minSingleSLHitsMax");
  minSingleSLHitsMin_ = pset.getUntrackedParameter<int>("minSingleSLHitsMin");
  allowedVariance_ = pset.getUntrackedParameter<int>("allowedVariance");
  allowDuplicates_ = pset.getUntrackedParameter<bool>("allowDuplicates");
  allowUncorrelatedPatterns_ = pset.getUntrackedParameter<bool>("allowUncorrelatedPatterns");
  minUncorrelatedHits_ = pset.getUntrackedParameter<int>("minUncorrelatedHits");
  saveOnPlace_ = pset.getUntrackedParameter<bool>("saveOnPlace");
  setLateralities_ = pset.getUntrackedParameter<bool>("setLateralities");
  if (debug_)
    LogDebug("PseudoBayesGrouping") << "PseudoBayesGrouping:: constructor";
}

PseudoBayesGrouping::~PseudoBayesGrouping() {
  if (debug_)
    LogDebug("PseudoBayesGrouping") << "PseudoBayesGrouping:: destructor";
  for (std::vector<DTPattern*>::iterator pat_it = allPatterns_.begin(); pat_it != allPatterns_.end(); pat_it++) {
    delete (*pat_it);
  }
}

// ============================================================================
// Main methods (initialise, run, finish)
// ============================================================================
void PseudoBayesGrouping::initialise(const edm::EventSetup& iEventSetup) {
  if (debug_)
    LogDebug("PseudoBayesGrouping") << "PseudoBayesGrouping::initialiase";
  if (debug_)
    LogDebug("PseudoBayesGrouping") << "PseudoBayesGrouping::initialiase using patterns file " << pattern_filename_;
  nPatterns_ = 0;
  //Load patterns from pattern root file with expected hits information
  TFile* f = TFile::Open(TString(pattern_filename_), "READ");
  std::vector<std::vector<std::vector<int>>>* pattern_reader =
      (std::vector<std::vector<std::vector<int>>>*)f->Get("allPatterns");
  for (std::vector<std::vector<std::vector<int>>>::iterator itPattern = (*pattern_reader).begin();
       itPattern != (*pattern_reader).end();
       ++itPattern) {
    //Loops over all patterns in the loop and constructs the Pattern object for each one
    LoadPattern(itPattern);
  }
  if (debug_)
    LogDebug("PseudoBayesGrouping") << "PseudoBayesGrouping::initialiase Total number of loaded patterns: "
                                    << nPatterns_;
  f->Close();
  delete f;

  prelimMatches_ = std::make_unique<CandidateGroupPtrs>();
  allMatches_ = std::make_unique<CandidateGroupPtrs>();
  finalMatches_ = std::make_unique<CandidateGroupPtrs>();
}

void PseudoBayesGrouping::LoadPattern(std::vector<std::vector<std::vector<int>>>::iterator itPattern) {
  if (debug_)
    LogDebug("PseudoBayesGrouping") << "PseudoBayesGrouping::LoadPattern Loading patterns seeded by: "
                                    << itPattern->at(0).at(0) << ", " << itPattern->at(0).at(1) << ", "
                                    << itPattern->at(0).at(2) << ", ";

  DTPattern p;
  //  for (auto itHits = itPattern->begin(); itHits != itPattern->end(); ++itHits) {
  bool is_seed = true;
  for (const auto& itHits : *itPattern) {
    //First entry is the seeding information
    if (is_seed) {
      p = DTPattern(itHits.at(0), itHits.at(1), itHits.at(2));
      is_seed = false;
    }
    //Other entries are the hits information
    else {
      if (itHits.begin() == itHits.end())
        continue;
      //We need to correct the geometry from pattern generation to reconstruction as they use slightly displaced basis
      else if (itHits.at(0) % 2 == 0) {
        p.addHit(std::make_tuple(itHits.at(0), itHits.at(1), itHits.at(2)));
      } else if (itHits.at(0) % 2 == 1) {
        p.addHit(std::make_tuple(itHits.at(0), itHits.at(1) - 1, itHits.at(2)));
      }
    }
  }
  //Classified by seeding layers for optimized search later
  //TODO::This can be vastly improved using std::bitset<8>, for example
  if (p.sl1() == 0) {
    if (p.sl2() == 7)
      L0L7Patterns_.push_back(&p);
    if (p.sl2() == 6)
      L0L6Patterns_.push_back(&p);
    if (p.sl2() == 5)
      L0L5Patterns_.push_back(&p);
    if (p.sl2() == 4)
      L0L4Patterns_.push_back(&p);
    if (p.sl2() == 3)
      L0L3Patterns_.push_back(&p);
    if (p.sl2() == 2)
      L0L2Patterns_.push_back(&p);
    if (p.sl2() == 1)
      L0L1Patterns_.push_back(&p);
  }
  if (p.sl1() == 1) {
    if (p.sl2() == 7)
      L1L7Patterns_.push_back(&p);
    if (p.sl2() == 6)
      L1L6Patterns_.push_back(&p);
    if (p.sl2() == 5)
      L1L5Patterns_.push_back(&p);
    if (p.sl2() == 4)
      L1L4Patterns_.push_back(&p);
    if (p.sl2() == 3)
      L1L3Patterns_.push_back(&p);
    if (p.sl2() == 2)
      L1L2Patterns_.push_back(&p);
  }
  if (p.sl1() == 2) {
    if (p.sl2() == 7)
      L2L7Patterns_.push_back(&p);
    if (p.sl2() == 6)
      L2L6Patterns_.push_back(&p);
    if (p.sl2() == 5)
      L2L5Patterns_.push_back(&p);
    if (p.sl2() == 4)
      L2L4Patterns_.push_back(&p);
    if (p.sl2() == 3)
      L2L3Patterns_.push_back(&p);
  }
  if (p.sl1() == 3) {
    if (p.sl2() == 7)
      L3L7Patterns_.push_back(&p);
    if (p.sl2() == 6)
      L3L6Patterns_.push_back(&p);
    if (p.sl2() == 5)
      L3L5Patterns_.push_back(&p);
    if (p.sl2() == 4)
      L3L4Patterns_.push_back(&p);
  }

  if (p.sl1() == 4) {
    if (p.sl2() == 7)
      L4L7Patterns_.push_back(&p);
    if (p.sl2() == 6)
      L4L6Patterns_.push_back(&p);
    if (p.sl2() == 5)
      L4L5Patterns_.push_back(&p);
  }
  if (p.sl1() == 5) {
    if (p.sl2() == 7)
      L5L7Patterns_.push_back(&p);
    if (p.sl2() == 6)
      L5L6Patterns_.push_back(&p);
  }
  if (p.sl1() == 6) {
    if (p.sl2() == 7)
      L6L7Patterns_.push_back(&p);
  }
  //Also creating a list of all patterns, needed later for deleting and avoid a memory leak
  allPatterns_.push_back(&p);
  nPatterns_++;
}

void PseudoBayesGrouping::run(Event& iEvent,
                              const EventSetup& iEventSetup,
                              const DTDigiCollection& digis,
                              MuonPathPtrs& mpaths) {
  //Takes dt digis collection and does the grouping for correlated hits, it is saved in a vector of up to 8 (or 4) correlated hits
  if (debug_)
    LogDebug("PseudoBayesGrouping") << "PseudoBayesGrouping::run";
  //Do initial cleaning
  CleanDigisByLayer();
  //Sort digis by layer
  FillDigisByLayer(&digis);
  //Sarch for patterns
  RecognisePatternsByLayerPairs();
  //Now sort patterns by qualities
  std::sort(prelimMatches_->begin(), prelimMatches_->end(), CandPointGreat());
  if (debug_ && !prelimMatches_->empty()) {
    LogDebug("PseudoBayesGrouping") << "PseudoBayesGrouping::run Pattern qualities before cleaning: ";
    for (const auto& cand_it : *prelimMatches_) {
      LogDebug("PseudoBayesGrouping") << cand_it->nLayerhits() << ", " << cand_it->nisGood() << ", " << cand_it->nhits()
                                      << ", " << cand_it->quality() << ", " << cand_it->candId();
    }
  }
  //And ghostbust patterns to retain higher quality ones
  ReCleanPatternsAndDigis();
  if (debug_)
    LogDebug("PseudoBayesGrouping") << "PseudoBayesGrouping::run Number of found patterns: " << finalMatches_->size();

  //Last organize candidates information into muonpaths to finalize the grouping
  FillMuonPaths(mpaths);
  if (debug_)
    LogDebug("PseudoBayesGrouping") << "PseudoBayesGrouping::run ended run";
}

void PseudoBayesGrouping::FillMuonPaths(MuonPathPtrs& mpaths) {
  //Loop over all selected candidates
  for (auto itCand = finalMatches_->begin(); itCand != finalMatches_->end(); itCand++) {
    if (debug_)
      LogDebug("PseudoBayesGrouping") << "PseudoBayesGrouping::run Create pointers ";
    DTPrimitivePtrs ptrPrimitive;
    for (int i = 0; i < 8; i++)
      ptrPrimitive.push_back(std::make_shared<DTPrimitive>());

    qualitybits qualityDTP;
    int intHit = 0;
    //And for each candidate loop over all grouped hits
    for (auto& itDTP : (*itCand)->candHits()) {
      if (debug_)
        LogDebug("PseudoBayesGrouping") << "PseudoBayesGrouping::run loop over dt hits to fill pointer";

      int layerHit = (*itDTP).layerId();
      //Back to the usual basis for SL
      if (layerHit >= 4) {
        (*itDTP).setLayerId(layerHit - 4);
      }
      qualitybits ref8Hit(std::pow(2, layerHit));
      //Get the predicted laterality
      if (setLateralities_) {
        int predLat = (*itCand)->pattern()->latHitIn(layerHit, (*itDTP).channelId(), allowedVariance_);
        if (predLat == -10 || predLat == 0) {
          (*itDTP).setLaterality(NONE);
        } else if (predLat == -1) {
          (*itDTP).setLaterality(LEFT);
        } else if (predLat == +1) {
          (*itDTP).setLaterality(RIGHT);
        }
      }
      //Only fill the DT primitives pointer if there is not one hit already in the layer
      if (qualityDTP != (qualityDTP | ref8Hit)) {
        if (debug_)
          LogDebug("PseudoBayesGrouping") << "PseudoBayesGrouping::run Adding hit to muon path";
        qualityDTP = (qualityDTP | ref8Hit);
        if (saveOnPlace_) {
          //This will save the primitive in a place of the vector equal to its L position
          ptrPrimitive.at(layerHit) = std::make_shared<DTPrimitive>((*itDTP));
        }
        if (!saveOnPlace_) {
          //This will save the primitive in order
          intHit++;
          ptrPrimitive.at(intHit) = std::make_shared<DTPrimitive>((*itDTP));
        }
      }
    }
    //Now, if there are empty spaces in the vector fill them full of daylight
    int ipow = 1;
    for (int i = 0; i <= 7; i++) {
      ipow *= 2;
      if (qualityDTP != (qualityDTP | qualitybits(1 << i))) {
        ptrPrimitive.at(i) = std::make_shared<DTPrimitive>();
      }
    }

    mpaths.emplace_back(
        std::make_shared<MuonPath>(ptrPrimitive, (short)(*itCand)->nLayerUp(), (short)(*itCand)->nLayerDown()));
  }
}

void PseudoBayesGrouping::RecognisePatternsByLayerPairs() {
  //Separated from main run function for clarity. Do all pattern recognition steps
  pidx_ = 0;
  //Compare L0-L7
  RecognisePatterns(digisinL0_, digisinL7_, L0L7Patterns_);
  //Compare L0-L6 and L1-L7
  RecognisePatterns(digisinL0_, digisinL6_, L0L6Patterns_);
  RecognisePatterns(digisinL1_, digisinL7_, L1L7Patterns_);
  //Compare L0-L5, L1-L6, L2-L7
  RecognisePatterns(digisinL0_, digisinL5_, L0L5Patterns_);
  RecognisePatterns(digisinL1_, digisinL6_, L1L6Patterns_);
  RecognisePatterns(digisinL2_, digisinL7_, L2L7Patterns_);
  //L0-L4, L1-L5, L2-L6, L3-L7
  RecognisePatterns(digisinL0_, digisinL4_, L0L4Patterns_);
  RecognisePatterns(digisinL1_, digisinL5_, L1L5Patterns_);
  RecognisePatterns(digisinL2_, digisinL6_, L2L6Patterns_);
  RecognisePatterns(digisinL3_, digisinL7_, L3L7Patterns_);
  //L1-L4, L2-L5, L3-L6
  RecognisePatterns(digisinL1_, digisinL4_, L1L4Patterns_);
  RecognisePatterns(digisinL2_, digisinL5_, L2L5Patterns_);
  RecognisePatterns(digisinL3_, digisinL6_, L3L6Patterns_);
  //L2-L4, L3-L5
  RecognisePatterns(digisinL2_, digisinL4_, L2L4Patterns_);
  RecognisePatterns(digisinL3_, digisinL5_, L3L5Patterns_);
  //L3-L4
  RecognisePatterns(digisinL3_, digisinL4_, L3L4Patterns_);
  //Uncorrelated SL1
  RecognisePatterns(digisinL0_, digisinL1_, L0L1Patterns_);
  RecognisePatterns(digisinL0_, digisinL2_, L0L2Patterns_);
  RecognisePatterns(digisinL0_, digisinL3_, L0L3Patterns_);
  RecognisePatterns(digisinL1_, digisinL2_, L1L2Patterns_);
  RecognisePatterns(digisinL1_, digisinL3_, L1L3Patterns_);
  RecognisePatterns(digisinL2_, digisinL3_, L2L3Patterns_);
  //Uncorrelated SL3
  RecognisePatterns(digisinL4_, digisinL5_, L4L5Patterns_);
  RecognisePatterns(digisinL4_, digisinL6_, L4L6Patterns_);
  RecognisePatterns(digisinL4_, digisinL7_, L4L7Patterns_);
  RecognisePatterns(digisinL5_, digisinL6_, L5L6Patterns_);
  RecognisePatterns(digisinL5_, digisinL7_, L5L7Patterns_);
  RecognisePatterns(digisinL6_, digisinL7_, L6L7Patterns_);
}

void PseudoBayesGrouping::RecognisePatterns(std::vector<DTPrimitive> digisinLDown,
                                            std::vector<DTPrimitive> digisinLUp,
                                            std::vector<DTPattern*> patterns) {
  //Loop over all hits and search for matching patterns (there will be four
  // amongst ~60, accounting for possible lateralities)
  for (auto dtPD_it = digisinLDown.begin(); dtPD_it != digisinLDown.end(); dtPD_it++) {
    int LDown = dtPD_it->layerId();
    int wireDown = dtPD_it->channelId();
    for (auto dtPU_it = digisinLUp.begin(); dtPU_it != digisinLUp.end(); dtPU_it++) {
      int LUp = dtPU_it->layerId();
      int wireUp = dtPU_it->channelId();
      int diff = wireUp - wireDown;
      for (auto pat_it = patterns.begin(); pat_it != patterns.end(); pat_it++) {
        //For each pair of hits in the layers search for the seeded patterns
        if ((*pat_it)->sl1() != (LDown) || (*pat_it)->sl2() != (LUp) || (*pat_it)->diff() != diff)
          continue;
        //If we are here a pattern was found and we can start comparing
        (*pat_it)->setHitDown(wireDown);
        auto cand = std::make_shared<CandidateGroup>(*pat_it);
        for (auto dtTest_it = alldigis_.begin(); dtTest_it != alldigis_.end(); dtTest_it++) {
          //Find hits matching to the pattern
          if (((*pat_it)->latHitIn(dtTest_it->layerId(), dtTest_it->channelId(), allowedVariance_)) != -999) {
            if (((*pat_it)->latHitIn(dtTest_it->layerId(), dtTest_it->channelId(), allowedVariance_)) == -10)
              cand->addHit((*dtTest_it), dtTest_it->layerId(), false);
            else
              cand->addHit((*dtTest_it), dtTest_it->layerId(), true);
          }
        }
        if ((cand->nhits() >= minNLayerHits_ &&
             (cand->nLayerUp() >= minSingleSLHitsMax_ || cand->nLayerDown() >= minSingleSLHitsMax_) &&
             (cand->nLayerUp() >= minSingleSLHitsMin_ && cand->nLayerDown() >= minSingleSLHitsMin_)) ||
            (allowUncorrelatedPatterns_ && ((cand->nLayerUp() >= minUncorrelatedHits_ && cand->nLayerDown() == 0) ||
                                            (cand->nLayerDown() >= minUncorrelatedHits_ && cand->nLayerUp() == 0)))) {
          if (debug_) {
            LogDebug("PseudoBayesGrouping") << "PseudoBayesGrouping::RecognisePatterns Pattern found for pair in "
                                            << LDown << " ," << wireDown << " ," << LUp << " ," << wireUp;
            LogDebug("PseudoBayesGrouping")
                << "Candidate has " << cand->nhits() << " hits with quality " << cand->quality();
            LogDebug("PseudoBayesGrouping") << *(*pat_it);
          }
          //We currently save everything at this level, might want to be more restrictive
          pidx_++;
          cand->setCandId(pidx_);
          prelimMatches_->push_back(std::move(cand));
          allMatches_->push_back(std::move(cand));
        }
      }
    }
  }
}

void PseudoBayesGrouping::FillDigisByLayer(const DTDigiCollection* digis) {
  //First we need to have separated lists of digis by layer
  if (debug_)
    LogDebug("PseudoBayesGrouping") << "PseudoBayesGrouping::FillDigisByLayer Classifying digis by layer";
  //  for (auto dtDigi_It = digis->begin(); dtDigi_It != digis->end(); dtDigi_It++) {
  for (const auto& dtDigi_It : *digis) {
    const DTLayerId dtLId = dtDigi_It.first;
    //Skip digis in SL theta which we are not interested on for the grouping
    for (auto digiIt = (dtDigi_It.second).first; digiIt != (dtDigi_It.second).second; digiIt++) {
      //Need to change notation slightly here
      if (dtLId.superlayer() == 2)
        continue;
      int layer = dtLId.layer() - 1;
      if (dtLId.superlayer() == 3)
        layer += 4;
      //Use the same format as for InitialGrouping to avoid tons of replicating classes, we will have some not used variables
      DTPrimitive dtpAux = DTPrimitive();
      dtpAux.setTDCTimeStamp(digiIt->time());
      dtpAux.setChannelId(digiIt->wire() - 1);
      dtpAux.setLayerId(layer);
      dtpAux.setSuperLayerId(dtLId.superlayer());
      dtpAux.setCameraId(dtLId.rawId());
      if (debug_)
        LogDebug("PseudoBayesGrouping") << "Hit in L " << layer << " SL " << dtLId.superlayer() << " WIRE "
                                        << digiIt->wire() - 1;
      if (layer == 0)
        digisinL0_.push_back(dtpAux);
      else if (layer == 1)
        digisinL1_.push_back(dtpAux);
      else if (layer == 2)
        digisinL2_.push_back(dtpAux);
      else if (layer == 3)
        digisinL3_.push_back(dtpAux);
      else if (layer == 4)
        digisinL4_.push_back(dtpAux);
      else if (layer == 5)
        digisinL5_.push_back(dtpAux);
      else if (layer == 6)
        digisinL6_.push_back(dtpAux);
      else if (layer == 7)
        digisinL7_.push_back(dtpAux);
      alldigis_.push_back(dtpAux);
    }
  }
}

void PseudoBayesGrouping::ReCleanPatternsAndDigis() {
  //GhostbustPatterns that share hits and are of lower quality
  if (prelimMatches_->empty()) {
    return;
  };
  while ((prelimMatches_->at(0)->nLayerhits() >= minNLayerHits_ &&
          (prelimMatches_->at(0)->nLayerUp() >= minSingleSLHitsMax_ ||
           prelimMatches_->at(0)->nLayerDown() >= minSingleSLHitsMax_) &&
          (prelimMatches_->at(0)->nLayerUp() >= minSingleSLHitsMin_ &&
           prelimMatches_->at(0)->nLayerDown() >= minSingleSLHitsMin_)) ||
         (allowUncorrelatedPatterns_ &&
          ((prelimMatches_->at(0)->nLayerUp() >= minUncorrelatedHits_ && prelimMatches_->at(0)->nLayerDown() == 0) ||
           (prelimMatches_->at(0)->nLayerDown() >= minUncorrelatedHits_ && prelimMatches_->at(0)->nLayerUp() == 0)))) {
    finalMatches_->push_back(prelimMatches_->at(0));
    auto itSel = finalMatches_->end() - 1;
    prelimMatches_->erase(prelimMatches_->begin());
    if (prelimMatches_->empty()) {
      return;
    };
    for (auto cand_it = prelimMatches_->begin(); cand_it != prelimMatches_->end(); cand_it++) {
      if (*(*cand_it) == *(*itSel) && allowDuplicates_)
        continue;
      for (const auto& dt_it : (*itSel)->candHits()) {  //.begin(); dt_it != (*itSel)->candHits().end(); dt_it++) {
        (*cand_it)->removeHit((*dt_it));
      }
    }

    std::sort(prelimMatches_->begin(), prelimMatches_->end(), CandPointGreat());
    if (debug_) {
      LogDebug("PseudoBayesGrouping") << "Pattern qualities: ";
      for (const auto& cand_it : *prelimMatches_) {
        LogDebug("PseudoBayesGrouping") << cand_it->nLayerhits() << ", " << cand_it->nisGood() << ", "
                                        << cand_it->nhits() << ", " << cand_it->quality() << ", " << cand_it->candId()
                                        << "\n";
      }
    }
  }
}

void PseudoBayesGrouping::CleanDigisByLayer() {
  digisinL0_.clear();
  digisinL1_.clear();
  digisinL2_.clear();
  digisinL3_.clear();
  digisinL4_.clear();
  digisinL5_.clear();
  digisinL6_.clear();
  digisinL7_.clear();
  alldigis_.clear();
  allMatches_->clear();
  prelimMatches_->clear();
  finalMatches_->clear();
}

void PseudoBayesGrouping::finish() {
  if (debug_)
    LogDebug("PseudoBayesGrouping") << "PseudoBayesGrouping: finish";
};
