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
  maxPathsPerMatch_ = pset.getUntrackedParameter<int>("maxPathsPerMatch");
  saveOnPlace_ = pset.getUntrackedParameter<bool>("saveOnPlace");
  setLateralities_ = pset.getUntrackedParameter<bool>("setLateralities");
  if (debug_)
    LogDebug("PseudoBayesGrouping") << "PseudoBayesGrouping:: constructor";
}

PseudoBayesGrouping::~PseudoBayesGrouping() {
  if (debug_)
    LogDebug("PseudoBayesGrouping") << "PseudoBayesGrouping:: destructor";
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

  TString patterns_folder = "L1Trigger/DTTriggerPhase2/data/";

  // Load all patterns
  // MB1
  LoadPattern(patterns_folder + "createdPatterns_MB1_left.root", 0, 0);
  LoadPattern(patterns_folder + "createdPatterns_MB1_right.root", 0, 2);
  // MB2
  LoadPattern(patterns_folder + "createdPatterns_MB2_left.root", 1, 0);
  LoadPattern(patterns_folder + "createdPatterns_MB2_right.root", 1, 2);
  // MB3
  LoadPattern(patterns_folder + "createdPatterns_MB3.root", 2, 1);
  // MB4
  LoadPattern(patterns_folder + "createdPatterns_MB4_left.root", 3, 0);
  LoadPattern(patterns_folder + "createdPatterns_MB4.root", 3, 1);
  LoadPattern(patterns_folder + "createdPatterns_MB4_right.root", 3, 2);

  prelimMatches_ = std::make_unique<CandidateGroupPtrs>();
  allMatches_ = std::make_unique<CandidateGroupPtrs>();
  finalMatches_ = std::make_unique<CandidateGroupPtrs>();
}

void PseudoBayesGrouping::LoadPattern(TString pattern_file_name, int MB_number, int SL_shift) {
  TFile* f = TFile::Open(pattern_file_name, "READ");

  std::vector<std::vector<std::vector<int>>>* pattern_reader =
      (std::vector<std::vector<std::vector<int>>>*)f->Get("allPatterns");

  for (std::vector<std::vector<std::vector<int>>>::iterator itPattern = (*pattern_reader).begin();
       itPattern != (*pattern_reader).end();
       ++itPattern) {
    if (debug_)
      LogDebug("PseudoBayesGrouping") << "PseudoBayesGrouping::LoadPattern Loading patterns seeded by: "
                                      << itPattern->at(0).at(0) << ", " << itPattern->at(0).at(1) << ", "
                                      << itPattern->at(0).at(2) << ", ";

    auto p = std::make_shared<DTPattern>();

    bool is_seed = true;
    for (const auto& itHits : *itPattern) {
      // First entry is the seeding information
      if (is_seed) {
        p = std::make_shared<DTPattern>(DTPattern(itHits.at(0), itHits.at(1), itHits.at(2)));
        is_seed = false;
      }
      // Other entries are the hits information
      else {
        if (itHits.begin() == itHits.end())
          continue;
        // We need to correct the geometry from pattern generation to reconstruction as they use slightly displaced basis
        else if (itHits.at(0) % 2 == 0) {
          p->addHit(std::make_tuple(itHits.at(0), itHits.at(1), itHits.at(2)));
        } else if (itHits.at(0) % 2 == 1) {
          p->addHit(std::make_tuple(itHits.at(0), itHits.at(1) - 1, itHits.at(2)));
        }
      }
    }
    // Classified by seeding layers for optimized search later
    // TODO::This can be vastly improved using std::bitset<8>, for example
    if (p->sl1() == 0) {
      if (p->sl2() == 7)
        L0L7Patterns_[MB_number][SL_shift].push_back(p);
      if (p->sl2() == 6)
        L0L6Patterns_[MB_number][SL_shift].push_back(p);
      if (p->sl2() == 5)
        L0L5Patterns_[MB_number][SL_shift].push_back(p);
      if (p->sl2() == 4)
        L0L4Patterns_[MB_number][SL_shift].push_back(p);
      if (p->sl2() == 3)
        L0L3Patterns_[MB_number][SL_shift].push_back(p);
      if (p->sl2() == 2)
        L0L2Patterns_[MB_number][SL_shift].push_back(p);
      if (p->sl2() == 1)
        L0L1Patterns_[MB_number][SL_shift].push_back(p);
    }
    if (p->sl1() == 1) {
      if (p->sl2() == 7)
        L1L7Patterns_[MB_number][SL_shift].push_back(p);
      if (p->sl2() == 6)
        L1L6Patterns_[MB_number][SL_shift].push_back(p);
      if (p->sl2() == 5)
        L1L5Patterns_[MB_number][SL_shift].push_back(p);
      if (p->sl2() == 4)
        L1L4Patterns_[MB_number][SL_shift].push_back(p);
      if (p->sl2() == 3)
        L1L3Patterns_[MB_number][SL_shift].push_back(p);
      if (p->sl2() == 2)
        L1L2Patterns_[MB_number][SL_shift].push_back(p);
    }
    if (p->sl1() == 2) {
      if (p->sl2() == 7)
        L2L7Patterns_[MB_number][SL_shift].push_back(p);
      if (p->sl2() == 6)
        L2L6Patterns_[MB_number][SL_shift].push_back(p);
      if (p->sl2() == 5)
        L2L5Patterns_[MB_number][SL_shift].push_back(p);
      if (p->sl2() == 4)
        L2L4Patterns_[MB_number][SL_shift].push_back(p);
      if (p->sl2() == 3)
        L2L3Patterns_[MB_number][SL_shift].push_back(p);
    }
    if (p->sl1() == 3) {
      if (p->sl2() == 7)
        L3L7Patterns_[MB_number][SL_shift].push_back(p);
      if (p->sl2() == 6)
        L3L6Patterns_[MB_number][SL_shift].push_back(p);
      if (p->sl2() == 5)
        L3L5Patterns_[MB_number][SL_shift].push_back(p);
      if (p->sl2() == 4)
        L3L4Patterns_[MB_number][SL_shift].push_back(p);
    }

    if (p->sl1() == 4) {
      if (p->sl2() == 7)
        L4L7Patterns_[MB_number][SL_shift].push_back(p);
      if (p->sl2() == 6)
        L4L6Patterns_[MB_number][SL_shift].push_back(p);
      if (p->sl2() == 5)
        L4L5Patterns_[MB_number][SL_shift].push_back(p);
    }
    if (p->sl1() == 5) {
      if (p->sl2() == 7)
        L5L7Patterns_[MB_number][SL_shift].push_back(p);
      if (p->sl2() == 6)
        L5L6Patterns_[MB_number][SL_shift].push_back(p);
    }
    if (p->sl1() == 6) {
      if (p->sl2() == 7)
        L6L7Patterns_[MB_number][SL_shift].push_back(p);
    }

    //Also creating a list of all patterns, needed later for deleting and avoid a memory leak
    allPatterns_[MB_number][SL_shift].push_back(p);
    nPatterns_++;
  }
  if (debug_)
    LogDebug("PseudoBayesGrouping") << "PseudoBayesGrouping::initialiase Total number of loaded patterns: "
                                    << nPatterns_;
  f->Close();
  delete f;
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
  DTChamberId chamber_id;
  // We just want the chamber ID of the first digi
  // as they are all the same --> create a loop and break it
  // after the first iteration
  for (const auto& detUnitIt : digis) {
    const DTLayerId layer_Id = detUnitIt.first;
    chamber_id = layer_Id.superlayerId().chamberId();
    break;
  }

  RecognisePatternsByLayerPairs(chamber_id);

  // Now sort patterns by qualities
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

    // Vector of all muon paths we may find
    std::vector<DTPrimitivePtrs> ptrPrimitive_vector;

    // We will have at least one muon path
    DTPrimitivePtrs ptrPrimitive;
    for (int i = 0; i < 8; i++)
      ptrPrimitive.push_back(std::make_shared<DTPrimitive>());

    ptrPrimitive_vector.push_back(ptrPrimitive);

    qualitybits qualityDTP;
    qualitybits qualityDTP2;
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

      // If one hit is already present in the current layer, for each ptrPrimitive already existing,
      // create a new with all its hits. Then, fill it with the new hit and add it to the primitives vector.
      // Do not consider more than 2 hits in the same wire or more than maxPathsPerMatch_ total muonpaths per finalMatches_
      if (qualityDTP == (qualityDTP | ref8Hit) && qualityDTP2 != (qualityDTP2 | ref8Hit) &&
          ptrPrimitive_vector.size() < maxPathsPerMatch_) {
        if (debug_)
          LogDebug("PseudoBayesGrouping") << "PseudoBayesGrouping::run Creating additional muon paths";

        qualityDTP2 = (qualityDTP2 | ref8Hit);

        int n_prim = ptrPrimitive_vector.size();

        for (int j = 0; j < n_prim; j++) {
          DTPrimitivePtrs tmpPrimitive;
          for (int i = 0; i < NUM_LAYERS_2SL; i++) {
            tmpPrimitive.push_back(ptrPrimitive_vector.at(j).at(i));
          }
          // Now save the hit in the new path
          if (saveOnPlace_) {
            //This will save the primitive in a place of the vector equal to its L position
            tmpPrimitive.at(layerHit) = std::make_shared<DTPrimitive>((*itDTP));
          }
          if (!saveOnPlace_) {
            //This will save the primitive in order
            tmpPrimitive.at(intHit) = std::make_shared<DTPrimitive>((*itDTP));
          }
          // Now add the new path to the vector of paths
          ptrPrimitive_vector.push_back(tmpPrimitive);
        }
      }

      // If there is not one hit already in the layer, fill the DT primitives pointers
      else {
        if (debug_)
          LogDebug("PseudoBayesGrouping") << "PseudoBayesGrouping::run Adding hit to muon path";

        qualityDTP = (qualityDTP | ref8Hit);

        // for (all paths --> fill them)
        for (auto prim_it = ptrPrimitive_vector.begin(); prim_it != ptrPrimitive_vector.end(); ++prim_it) {
          if (saveOnPlace_) {
            //This will save the primitive in a place of the vector equal to its L position
            prim_it->at(layerHit) = std::make_shared<DTPrimitive>((*itDTP));
          }
          if (!saveOnPlace_) {
            //This will save the primitive in order
            intHit++;
            prim_it->at(intHit) = std::make_shared<DTPrimitive>((*itDTP));
          }
        }
      }
    }

    stringstream ss;

    int n_paths = ptrPrimitive_vector.size();

    for (int n_path = 0; n_path < n_paths; ++n_path) {
      mpaths.emplace_back(std::make_shared<MuonPath>(
          ptrPrimitive_vector.at(n_path), (short)(*itCand)->nLayerUp(), (short)(*itCand)->nLayerDown()));
    }
  }
}

void PseudoBayesGrouping::RecognisePatternsByLayerPairs(DTChamberId chamber_ID) {
  // chamber_ID traslated to MB, wheel, sector
  int MB = chamber_ID.station() - 1;
  int wheel = chamber_ID.wheel();
  int sector = chamber_ID.sector();

  // shift of SL3 wrt SL1
  int shift = -1;

  // Now define DT geometry depending on its ID

  // MB1
  if (MB == 0) {
    if (wheel == -1 || wheel == -2)
      shift = 2;  // positive (right)
    else if (wheel == 1 || wheel == 2)
      shift = 0;  // negative (left)
    else if (wheel == 0) {
      if (sector == 1 || sector == 4 || sector == 5 || sector == 8 || sector == 9 || sector == 12)
        shift = 2;  // positive (right)
      else
        shift = 0;  // negative (left)
    }
  }
  // MB2
  else if (MB == 1) {
    if (wheel == -1 || wheel == -2)
      shift = 0;  // negative (left)
    else if (wheel == 1 || wheel == 2)
      shift = 2;  // positive (right)
    else if (wheel == 0) {
      if (sector == 1 || sector == 4 || sector == 5 || sector == 8 || sector == 9 || sector == 12)
        shift = 0;  // negative (left)
      else
        shift = 2;  // positive (right)
    }
  }
  // MB3
  else if (MB == 2) {
    shift = 1;  // shift is always 0 in MB3
  }
  // MB4
  else if (MB == 3) {
    if (wheel == -1 || wheel == -2)
      if (sector == 4 || sector == 9 || sector == 11 || sector == 13)
        shift = 1;  // no shift
      else if (sector == 5 || sector == 6 || sector == 7 || sector == 8 || sector == 14)
        shift = 2;  // positive (right)
      else
        shift = 0;  // negative (left)
    else if (wheel == 1 || wheel == 2)
      if (sector == 4 || sector == 9 || sector == 11 || sector == 13)
        shift = 1;  // no shift
      else if (sector == 1 || sector == 2 || sector == 3 || sector == 10 || sector == 12)
        shift = 2;  // positive (right)
      else
        shift = 0;  // negative (left)
    else if (wheel == 0)
      if (sector == 4 || sector == 9 || sector == 11 || sector == 13)
        shift = 1;  // no shift
      else if (sector == 2 || sector == 3 || sector == 5 || sector == 8 || sector == 10)
        shift = 2;  // positive (right)
      else
        shift = 0;  // negative (left)
    else
      return;
  }

  //Separated from main run function for clarity. Do all pattern recognition steps
  pidx_ = 0;
  //Compare L0-L7
  RecognisePatterns(digisinL0_, digisinL7_, L0L7Patterns_[MB][shift]);
  //Compare L0-L6 and L1-L7
  RecognisePatterns(digisinL0_, digisinL6_, L0L6Patterns_[MB][shift]);
  RecognisePatterns(digisinL1_, digisinL7_, L1L7Patterns_[MB][shift]);
  //Compare L0-L5, L1-L6, L2-L7
  RecognisePatterns(digisinL0_, digisinL5_, L0L5Patterns_[MB][shift]);
  RecognisePatterns(digisinL1_, digisinL6_, L1L6Patterns_[MB][shift]);
  RecognisePatterns(digisinL2_, digisinL7_, L2L7Patterns_[MB][shift]);
  //L0-L4, L1-L5, L2-L6, L3-L7
  RecognisePatterns(digisinL0_, digisinL4_, L0L4Patterns_[MB][shift]);
  RecognisePatterns(digisinL1_, digisinL5_, L1L5Patterns_[MB][shift]);
  RecognisePatterns(digisinL2_, digisinL6_, L2L6Patterns_[MB][shift]);
  RecognisePatterns(digisinL3_, digisinL7_, L3L7Patterns_[MB][shift]);
  //L1-L4, L2-L5, L3-L6
  RecognisePatterns(digisinL1_, digisinL4_, L1L4Patterns_[MB][shift]);
  RecognisePatterns(digisinL2_, digisinL5_, L2L5Patterns_[MB][shift]);
  RecognisePatterns(digisinL3_, digisinL6_, L3L6Patterns_[MB][shift]);
  //L2-L4, L3-L5
  RecognisePatterns(digisinL2_, digisinL4_, L2L4Patterns_[MB][shift]);
  RecognisePatterns(digisinL3_, digisinL5_, L3L5Patterns_[MB][shift]);
  //L3-L4
  RecognisePatterns(digisinL3_, digisinL4_, L3L4Patterns_[MB][shift]);
  //Uncorrelated SL1
  RecognisePatterns(digisinL0_, digisinL1_, L0L1Patterns_[MB][shift]);
  RecognisePatterns(digisinL0_, digisinL2_, L0L2Patterns_[MB][shift]);
  RecognisePatterns(digisinL0_, digisinL3_, L0L3Patterns_[MB][shift]);
  RecognisePatterns(digisinL1_, digisinL2_, L1L2Patterns_[MB][shift]);
  RecognisePatterns(digisinL1_, digisinL3_, L1L3Patterns_[MB][shift]);
  RecognisePatterns(digisinL2_, digisinL3_, L2L3Patterns_[MB][shift]);
  //Uncorrelated SL3
  RecognisePatterns(digisinL4_, digisinL5_, L4L5Patterns_[MB][shift]);
  RecognisePatterns(digisinL4_, digisinL6_, L4L6Patterns_[MB][shift]);
  RecognisePatterns(digisinL4_, digisinL7_, L4L7Patterns_[MB][shift]);
  RecognisePatterns(digisinL5_, digisinL6_, L5L6Patterns_[MB][shift]);
  RecognisePatterns(digisinL5_, digisinL7_, L5L7Patterns_[MB][shift]);
  RecognisePatterns(digisinL6_, digisinL7_, L6L7Patterns_[MB][shift]);
}

void PseudoBayesGrouping::RecognisePatterns(std::vector<DTPrimitive> digisinLDown,
                                            std::vector<DTPrimitive> digisinLUp,
                                            DTPatternPtrs patterns) {
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
            LogDebug("PseudoBayesGrouping")
                << "PseudoBayesGrouping::RecognisePatterns Pattern found for pair in " << LDown << " ," << wireDown
                << " ," << LUp << " ," << wireUp << "\n"
                << "Candidate has " << cand->nhits() << " hits with quality " << cand->quality() << "\n"
                << *(*pat_it);
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
      for (const auto& dt_it : (*itSel)->candHits()) {
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
