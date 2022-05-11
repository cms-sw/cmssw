#include "L1Trigger/DTTriggerPhase2/interface/MPQualityEnhancerFilter.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;
using namespace std;
using namespace cmsdt;
;

// ============================================================================
// Constructors and destructor
// ============================================================================
MPQualityEnhancerFilter::MPQualityEnhancerFilter(const ParameterSet &pset)
    : MPFilter(pset), debug_(pset.getUntrackedParameter<bool>("debug")) {}

// ============================================================================
// Main methods (initialise, run, finish)
// ============================================================================
void MPQualityEnhancerFilter::initialise(const edm::EventSetup &iEventSetup) {}

void MPQualityEnhancerFilter::run(edm::Event &iEvent,
                                  const edm::EventSetup &iEventSetup,
                                  std::vector<metaPrimitive> &inMPaths,
                                  std::vector<metaPrimitive> &outMPaths) {
  std::vector<metaPrimitive> buff;
  std::vector<metaPrimitive> buff2;

  filterCousins(inMPaths, buff);
  if (debug_) {
    LogDebug("MPQualityEnhancerFilter") << "Ended Cousins Filter. The final primitives before Refiltering are: ";
    for (unsigned int i = 0; i < buff.size(); i++) {
      printmP(buff[i]);
    }
    LogDebug("MPQualityEnhancerFilter") << "Total Primitives = " << buff.size();
  }
  refilteringCousins(buff, buff2);
  if (debug_) {
    LogDebug("MPQualityEnhancerFilter") << "Ended Cousins Refilter. The final primitives before UniqueFilter are: ";
    for (unsigned int i = 0; i < buff2.size(); i++) {
      printmP(buff2[i]);
    }
    LogDebug("MPQualityEnhancerFilter") << "Total Primitives = " << buff2.size();
  }
  filterUnique(buff2, outMPaths);

  if (debug_) {
    LogDebug("MPQualityEnhancerFilter") << "Ended Unique Filter. The final primitives are: ";
    for (unsigned int i = 0; i < outMPaths.size(); i++) {
      printmP(outMPaths[i]);
      LogDebug("MPQualityEnhancerFilter");
    }
    LogDebug("MPQualityEnhancerFilter") << "Total Primitives = " << outMPaths.size();
  }

  buff.clear();
  buff2.clear();
}

void MPQualityEnhancerFilter::finish(){};

///////////////////////////
///  OTHER METHODS
int MPQualityEnhancerFilter::areCousins(metaPrimitive mp, metaPrimitive second_mp) {
  if (mp.rawId != second_mp.rawId)
    return 0;
  if (mp.wi1 == second_mp.wi1 and mp.wi1 != -1 and mp.tdc1 != -1)
    return 1;
  if (mp.wi2 == second_mp.wi2 and mp.wi2 != -1 and mp.tdc2 != -1)
    return 2;
  if (mp.wi3 == second_mp.wi3 and mp.wi3 != -1 and mp.tdc3 != -1)
    return 3;
  if (mp.wi4 == second_mp.wi4 and mp.wi4 != -1 and mp.tdc4 != -1)
    return 4;
  return 0;
}

int MPQualityEnhancerFilter::rango(metaPrimitive mp) {
  if (mp.quality == 1 or mp.quality == 2)
    return 3;
  if (mp.quality == 3 or mp.quality == 4)
    return 4;
  return 0;
}

void MPQualityEnhancerFilter::filterCousins(std::vector<metaPrimitive> &inMPaths,
                                            std::vector<metaPrimitive> &outMPaths) {
  int primo_index = 0;
  bool oneof4 = false;
  int bestI = -1;
  double bestChi2 = 9999;
  if (inMPaths.size() == 1) {
    if (debug_) {
      printmP(inMPaths[0]);
    }
    outMPaths.push_back(inMPaths[0]);
  } else if (inMPaths.size() > 1) {
    for (int i = 0; i < int(inMPaths.size()); i++) {
      if (debug_) {
        printmP(inMPaths[i]);
      }
      if (areCousins(inMPaths[i], inMPaths[i - primo_index]) == 0) {
        if (oneof4) {
          outMPaths.push_back(inMPaths[bestI]);

          bestI = -1;
          bestChi2 = 9999;
          oneof4 = false;
        } else {
          for (int j = i - primo_index; j < i; j++) {
            outMPaths.push_back(inMPaths[j]);
          }
        }
        i--;
        primo_index = 0;
        continue;
      }
      if (rango(inMPaths[i]) == 4) {
        oneof4 = true;
        if (bestChi2 > inMPaths[i].chi2) {
          bestChi2 = inMPaths[i].chi2;
          bestI = i;
        }
      }
      if (areCousins(inMPaths[i], inMPaths[i + 1]) != 0) {
        primo_index++;
      } else {  //areCousing==0
        if (oneof4) {
          outMPaths.push_back(inMPaths[bestI]);
          bestI = -1;
          bestChi2 = 9999;
          oneof4 = false;
        } else {
          for (int j = i - primo_index; j <= i; j++) {
            outMPaths.push_back(inMPaths[j]);
          }
        }
        primo_index = 0;
      }
    }
  }

}  //End filterCousins

void MPQualityEnhancerFilter::refilteringCousins(std::vector<metaPrimitive> &inMPaths,
                                                 std::vector<metaPrimitive> &outMPaths) {
  if (debug_)
    LogDebug("MPQualityEnhancerFilter") << "filtering: starting cousins refiltering\n";
  int bestI = -1;
  double bestChi2 = 9999;
  bool oneOf4 = false;
  int back = 0;

  if (inMPaths.size() > 1) {
    for (unsigned int i = 0; i < inMPaths.size(); i++) {
      if (debug_) {
        LogDebug("MPQualityEnhancerFilter") << "filtering: starting with mp " << i << ": ";
        printmP(inMPaths[i]);
        LogDebug("MPQualityEnhancerFilter");
      }
      if (rango(inMPaths[i]) == 4 && bestChi2 > inMPaths[i].chi2) {  // 4h prim with a smaller chi2
        if (debug_) {
          LogDebug("MPQualityEnhancerFilter") << "filtering: mp " << i << " is the best 4h primitive";
        }
        oneOf4 = true;
        bestI = i;
        bestChi2 = inMPaths[i].chi2;
      }
      if (i == inMPaths.size() - 1) {  //You can't compare the last one with the next one
        if (oneOf4) {
          outMPaths.push_back(inMPaths[bestI]);
        } else {
          for (unsigned int j = i - back; j <= i; j++) {
            outMPaths.push_back(inMPaths[j]);
          }
        }
      } else {
        if (areCousins(inMPaths[i], inMPaths[i + 1]) == 0) {  //they arent cousins
          if (debug_) {
            LogDebug("MPQualityEnhancerFilter") << "mp " << i << " and mp " << i + 1 << " are not cousins";
          }
          if (oneOf4) {
            outMPaths.push_back(inMPaths[bestI]);
            if (debug_)
              LogDebug("MPQualityEnhancerFilter") << "kept4 mp " << bestI;
            oneOf4 = false;  //reset 4h variables
            bestI = -1;
            bestChi2 = 9999;
          } else {
            for (unsigned int j = i - back; j <= i; j++) {
              outMPaths.push_back(inMPaths[j]);
              if (debug_)
                LogDebug("MPQualityEnhancerFilter") << "kept3 mp " << j;
            }
          }
          back = 0;
        } else {  // they are cousins
          back++;
        }
      }
    }
  } else if (inMPaths.size() == 1) {
    outMPaths.push_back(inMPaths[0]);
  }
}

void MPQualityEnhancerFilter::filterUnique(std::vector<metaPrimitive> &inMPaths,
                                           std::vector<metaPrimitive> &outMPaths) {
  constexpr double xTh = 0.001;
  constexpr double tPhiTh = 0.001;
  constexpr double t0Th = 0.001;
  for (size_t i = 0; i < inMPaths.size(); i++) {
    bool visto = false;
    for (size_t j = i + 1; j < inMPaths.size(); j++) {
      if ((std::abs(inMPaths[i].x - inMPaths[j].x) <= xTh) &&
          (std::abs(inMPaths[i].tanPhi - inMPaths[j].tanPhi) <= tPhiTh) &&
          (std::abs(inMPaths[i].t0 - inMPaths[j].t0) <= t0Th))
        visto = true;
    }
    if (!visto)
      outMPaths.push_back(inMPaths[i]);
  }
}

void MPQualityEnhancerFilter::printmP(metaPrimitive mP) {
  DTSuperLayerId slId(mP.rawId);
  LogDebug("MPQualityEnhancerFilter") << slId << "\t"
                                      << " " << setw(2) << left << mP.wi1 << " " << setw(2) << left << mP.wi2 << " "
                                      << setw(2) << left << mP.wi3 << " " << setw(2) << left << mP.wi4 << " " << setw(5)
                                      << left << mP.tdc1 << " " << setw(5) << left << mP.tdc2 << " " << setw(5) << left
                                      << mP.tdc3 << " " << setw(5) << left << mP.tdc4 << " " << setw(10) << right
                                      << mP.x << " " << setw(9) << left << mP.tanPhi << " " << setw(5) << left << mP.t0
                                      << " " << setw(13) << left << mP.chi2 << " r:" << rango(mP);
}
