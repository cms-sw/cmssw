#include "L1Trigger/DTTriggerPhase2/interface/MPQualityEnhancerFilterBayes.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;
using namespace std;
using namespace cmsdt;
;

// ============================================================================
// Constructors and destructor
// ============================================================================
MPQualityEnhancerFilterBayes::MPQualityEnhancerFilterBayes(const ParameterSet &pset)
    : MPFilter(pset), debug_(pset.getUntrackedParameter<bool>("debug")) {}

// ============================================================================
// Main methods (initialise, run, finish)
// ============================================================================
void MPQualityEnhancerFilterBayes::initialise(const edm::EventSetup &iEventSetup) {}

void MPQualityEnhancerFilterBayes::run(edm::Event &iEvent,
                                       const edm::EventSetup &iEventSetup,
                                       std::vector<metaPrimitive> &inMPaths,
                                       std::vector<metaPrimitive> &outMPaths) {
  filterCousins(inMPaths, outMPaths);
  if (debug_) {
    LogDebug("MPQualityEnhancerFilterBayes") << "Ended Cousins Filter. The final primitives before Refiltering are: ";
    for (unsigned int i = 0; i < outMPaths.size(); i++) {
      printmP(outMPaths[i]);
    }
    LogDebug("MPQualityEnhancerFilterBayes") << "Total Primitives = " << outMPaths.size();
  }
}

void MPQualityEnhancerFilterBayes::finish(){};

///////////////////////////
///  OTHER METHODS
int MPQualityEnhancerFilterBayes::areCousins(metaPrimitive mp, metaPrimitive second_mp) {
  DTSuperLayerId mpId(mp.rawId);
  DTSuperLayerId second_mpId(second_mp.rawId);

  int output = 0;
  if (mpId.wheel() != second_mpId.wheel() || mpId.station() != second_mpId.station() ||
      mpId.sector() != second_mpId.sector()) {
    return output;
  }

  if (mp.wi1 == second_mp.wi1 and mp.tdc1 == second_mp.tdc1 and mp.wi1 != -1 and mp.tdc1 != -1)
    output = 1;
  else if (mp.wi2 == second_mp.wi2 and mp.tdc2 == second_mp.tdc2 and mp.wi2 != -1 and mp.tdc2 != -1)
    output = 2;
  else if (mp.wi3 == second_mp.wi3 and mp.tdc3 == second_mp.tdc3 and mp.wi3 != -1 and mp.tdc3 != -1)
    output = 3;
  else if (mp.wi4 == second_mp.wi4 and mp.tdc4 == second_mp.tdc4 and mp.wi4 != -1 and mp.tdc4 != -1)
    output = 4;
  else if (mp.wi5 == second_mp.wi5 and mp.tdc5 == second_mp.tdc5 and mp.wi5 != -1 and mp.tdc5 != -1)
    output = 5;
  else if (mp.wi6 == second_mp.wi6 and mp.tdc6 == second_mp.tdc6 and mp.wi6 != -1 and mp.tdc6 != -1)
    output = 6;
  else if (mp.wi7 == second_mp.wi7 and mp.tdc7 == second_mp.tdc7 and mp.wi7 != -1 and mp.tdc7 != -1)
    output = 7;
  else if (mp.wi8 == second_mp.wi8 and mp.tdc8 == second_mp.tdc8 and mp.wi8 != -1 and mp.tdc8 != -1)
    output = 8;

  return output;
}

///////////////////////////
///  OTHER METHODS
bool MPQualityEnhancerFilterBayes::areSame(metaPrimitive mp, metaPrimitive second_mp) {
  if (mp.rawId != second_mp.rawId)
    return false;
  if (mp.wi1 != second_mp.wi1 or mp.tdc1 != second_mp.tdc1)
    return false;
  if (mp.wi2 != second_mp.wi2 or mp.tdc2 != second_mp.tdc2)
    return false;
  if (mp.wi3 != second_mp.wi3 or mp.tdc3 != second_mp.tdc3)
    return false;
  if (mp.wi4 != second_mp.wi4 or mp.tdc4 != second_mp.tdc4)
    return false;
  if (mp.wi5 != second_mp.wi5 or mp.tdc5 != second_mp.tdc5)
    return false;
  if (mp.wi6 != second_mp.wi6 or mp.tdc6 != second_mp.tdc6)
    return false;
  if (mp.wi7 != second_mp.wi7 or mp.tdc7 != second_mp.tdc7)
    return false;
  if (mp.wi8 != second_mp.wi8 or mp.tdc8 != second_mp.tdc8)
    return false;
  return true;
}

int MPQualityEnhancerFilterBayes::shareSL(metaPrimitive mp, metaPrimitive second_mp) {
  DTSuperLayerId mpId(mp.rawId);
  DTSuperLayerId second_mpId(second_mp.rawId);

  int output = 0;
  if (mpId.wheel() != second_mpId.wheel() || mpId.station() != second_mpId.station() ||
      mpId.sector() != second_mpId.sector()) {
    return output;
  }

  int SL1 = 0;
  int SL3 = 0;

  int SL1_shared = 0;
  int SL3_shared = 0;

  if (mp.wi1 != -1 and mp.tdc1 != -1) {
    ++SL1;
    if (mp.wi1 == second_mp.wi1 and mp.tdc1 == second_mp.tdc1) {
      ++SL1_shared;
    }
  }
  if (mp.wi2 != -1 and mp.tdc2 != -1) {
    ++SL1;
    if (mp.wi2 == second_mp.wi2 and mp.tdc2 == second_mp.tdc2) {
      ++SL1_shared;
    }
  }
  if (mp.wi3 != -1 and mp.tdc3 != -1) {
    ++SL1;
    if (mp.wi3 == second_mp.wi3 and mp.tdc3 == second_mp.tdc3) {
      ++SL1_shared;
    }
  }
  if (mp.wi4 != -1 and mp.tdc4 != -1) {
    ++SL1;
    if (mp.wi4 == second_mp.wi4 and mp.tdc4 == second_mp.tdc4) {
      ++SL1_shared;
    }
  }

  if (mp.wi5 != -1 and mp.tdc5 != -1) {
    ++SL3;
    if (mp.wi5 == second_mp.wi5 and mp.tdc5 == second_mp.tdc5) {
      ++SL3_shared;
    }
  }
  if (mp.wi6 != -1 and mp.tdc6 != -1) {
    ++SL3;
    if (mp.wi6 == second_mp.wi6 and mp.tdc6 == second_mp.tdc6) {
      ++SL3_shared;
    }
  }
  if (mp.wi7 != -1 and mp.tdc7 != -1) {
    ++SL3;
    if (mp.wi7 == second_mp.wi7 and mp.tdc7 == second_mp.tdc7) {
      ++SL3_shared;
    }
  }
  if (mp.wi8 != -1 and mp.tdc8 != -1) {
    ++SL3;
    if (mp.wi8 == second_mp.wi8 and mp.tdc8 == second_mp.tdc8) {
      ++SL3_shared;
    }
  }

  // If the two mp share all hits in a SL, we consider that they share that SL
  if (SL1_shared == SL1 || SL3_shared == SL3)
    output = 1;

  return output;
}

int MPQualityEnhancerFilterBayes::BX(metaPrimitive mp) {
  int bx;
  bx = (int)round(mp.t0 / (float)LHC_CLK_FREQ);
  return bx;
}

// Is this really needed?
int MPQualityEnhancerFilterBayes::rango(metaPrimitive mp) {
  // Correlated
  if (mp.quality > 5)
    return 2;
  // uncorrelated
  else
    return 1;
}

void MPQualityEnhancerFilterBayes::filterCousins(std::vector<metaPrimitive> &inMPaths,
                                                 std::vector<metaPrimitive> &outMPaths) {
  // At the beginning, we want to keep all mpaths
  std::vector<bool> keep_this(inMPaths.size(), true);

  // If we have just one mpath, save it
  if (inMPaths.size() == 1) {
    if (debug_) {
      printmP(inMPaths[0]);
    }
    outMPaths.push_back(inMPaths[0]);
  }
  // More than one mpath
  else if (inMPaths.size() > 1) {
    for (int i = 0; i < int(inMPaths.size()); i++) {
      if (debug_) {
        printmP(inMPaths[i]);
      }
      // If we have already decided to reject the candidate, skip it
      if (keep_this[i] == false)
        continue;
      for (int j = i + 1; j < int(inMPaths.size()); j++) {
        // If we have already decided to reject the candidate, skip it
        if (keep_this[j] == false)
          continue;
        // Case they are the same, keep the first one
        if (areSame(inMPaths[i], inMPaths[j]) == true)
          keep_this[i] = false;
        // Case they are cousins, keep the best one
        if (areCousins(inMPaths[i], inMPaths[j]) != 0) {
          // In case both are correlated, they have to share a full SL
          if (inMPaths[i].quality > 5 && inMPaths[j].quality > 5 && shareSL(inMPaths[i], inMPaths[j]) == 0) {
            continue;
          }

          // Compare only if rango is the same (both correlated or both not-correlated)
          // if (rango(inMPaths[i]) != rango(inMPaths[j])) continue;

          // If rango is the same, keep higher quality one
          // Still, keep lower-quality one if it has lower Chi2
          // and if its BX is different to the higher-quality one
          if (inMPaths[i].quality > inMPaths[j].quality) {
            if ((inMPaths[i].chi2 < inMPaths[j].chi2) || BX(inMPaths[i]) == BX(inMPaths[j]))
              keep_this[j] = false;
          } else if (inMPaths[i].quality < inMPaths[j].quality) {
            if ((inMPaths[i].chi2 > inMPaths[j].chi2) || BX(inMPaths[i]) == BX(inMPaths[j]))
              keep_this[i] = false;
          } else {  // if they have same quality
            // If quality is 8, keep both
            // if (inMPaths[i].quality >= 8) continue;
            // Otherwise, keep the one with better Chi2
            // and also the one with worse Chi2 if its BX is different
            // else{
            if (inMPaths[i].chi2 > inMPaths[j].chi2 && BX(inMPaths[i]) == BX(inMPaths[j]))
              keep_this[i] = false;
            else if (inMPaths[i].chi2 < inMPaths[j].chi2 && BX(inMPaths[i]) == BX(inMPaths[j]))
              keep_this[j] = false;
            else
              continue;
            //}
          }
        }
      }
    }
    // Finally, fill the output with accepted candidates
    for (int i = 0; i < int(inMPaths.size()); i++)
      if (keep_this[i] == true)
        outMPaths.push_back(inMPaths[i]);
  }
}

void MPQualityEnhancerFilterBayes::printmP(metaPrimitive mP) {
  DTSuperLayerId slId(mP.rawId);
  LogDebug("MPQualityEnhancerFilterBayes")
      << slId << "\t"
      << " " << setw(2) << left << mP.wi1 << " " << setw(2) << left << mP.wi2 << " " << setw(2) << left << mP.wi3 << " "
      << setw(2) << left << mP.wi4 << " " << setw(5) << left << mP.tdc1 << " " << setw(5) << left << mP.tdc2 << " "
      << setw(5) << left << mP.tdc3 << " " << setw(5) << left << mP.tdc4 << " " << setw(10) << right << mP.x << " "
      << setw(9) << left << mP.tanPhi << " " << setw(5) << left << mP.t0 << " " << setw(13) << left << mP.chi2
      << " r:" << rango(mP);
}
