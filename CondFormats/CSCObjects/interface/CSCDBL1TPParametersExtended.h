#ifndef CSCObjects_CSCDBL1TPParametersExtended_h
#define CSCObjects_CSCDBL1TPParametersExtended_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>

/** \class CSCDBL1TPParametersExtended
 *  \author Sven Dildick
 *
 * Description: Configuration parameters needed for the Phase-1 upgrade of the level-1
 *              CSC trigger primitives emulator. ME1/1 ALCT, CLCT and TMB have own members.
 *
 */

union S {
  unsigned int i;
  bool b;
};

class CSCDBL1TPParametersExtended {
public:
  CSCDBL1TPParametersExtended();
  ~CSCDBL1TPParametersExtended();

  int getValueInt(const std::string&) const;
  bool getValueBool(const std::string&) const;

  void setValue(const std::string&, int);
  void setValue(const std::string&, bool);

private:
  std::vector<int> paramsInt_;
  std::vector<int> paramsBool_;

  const std::vector<std::string> paramNamesBool_{
      /* ME11 ALCT Phase-1 Upgrade Parameters */
      "me11_phase1_alctNarrowMaskForR1a",
      "me11_phase1_alctGhostCancellationSideQuality",
      "me11_phase1_alctUseCorrectedBx",

      /* ME11 CLCT Phase-1 Upgrade Parameters */
      "me11_phase1_useDeadTimeZoning",
      "me11_phase1_useDynamicStateMachineZone",
      "me11_phase1_clctUseCorrectedBx",

      /* ME11 TMB Phase-1 Upgrade Parameters */
      "me11_phase1_tmbReadoutEarliest2",
      "me11_phase1_tmbDropUsedAlcts",
      "me11_phase1_clctToAlct",
      "me11_phase1_tmbDropUsedClcts",
      "me11_phase1_matchEarliestAlctME11Only",
      "me11_phase1_matchEarliestClctME11Only",
  };

  const std::vector<std::string> paramNamesInt_{
      /* Parameters %for 2007 version of ALCT firmware */
      "alctFifoTbins",
      "alctFifoPretrig",
      "alctDriftDelay",
      "alctNplanesHitPretrig",
      "alctNplanesHitPattern",
      "alctNplanesHitAccelPretrig",
      "alctNplanesHitAccelPattern",
      "alctTrigMode",
      "alctAccelMode",
      "alctL1aWindowWidth",
      "alctEarlyTbins",

      /* Parameters for 2007 version of CLCT firmware */
      "clctFifoTbins",
      "clctFifoPretrig",
      "clctHitPersist",
      "clctDriftDelay",
      "clctNplanesHitPretrig",
      "clctNplanesHitPattern",
      "clctPidThreshPretrig",
      "clctMinSeparation",

      /* Parameters for 2007 version of TMB firmware */
      "tmbMpcBlockMe1a",
      "tmbAlctTrigEnable",
      "tmbClctTrigEnable",
      "tmbMatchTrigEnable",
      "tmbMatchTrigWindowSize",
      "tmbTmbL1aWindowSize",

      /* ME11 ALCT Phase-1 Upgrade Parameters */
      "me11_phase1_alctFifoTbins",
      "me11_phase1_alctFifoPretrig",
      "me11_phase1_alctDriftDelay",
      "me11_phase1_alctNplanesHitPretrig",
      "me11_phase1_alctNplanesHitPattern",
      "me11_phase1_alctNplanesHitAccelPretrig",
      "me11_phase1_alctNplanesHitAccelPattern",
      "me11_phase1_alctTrigMode",
      "me11_phase1_alctAccelMode",
      "me11_phase1_alctL1aWindowWidth",
      "me11_phase1_alctEarlyTbins",
      "me11_phase1_alctHitPersist",
      "me11_phase1_alctGhostCancellationBxDepth",
      "me11_phase1_alctPretrigDeadtime",

      /* ME11 CLCT Phase-1 Upgrade Parameters */
      "me11_phase1_clctFifoTbins",
      "me11_phase1_clctFifoPretrig",
      "me11_phase1_clctHitPersist",
      "me11_phase1_clctDriftDelay",
      "me11_phase1_clctNplanesHitPretrig",
      "me11_phase1_clctNplanesHitPattern",
      "me11_phase1_clctPidThreshPretrig",
      "me11_phase1_clctMinSeparation",
      "me11_phase1_clctStartBxShift",
      "me11_phase1_clctStateMachineZone",
      "me11_phase1_clctPretriggerTriggerZone",

      /* ME11 TMB Phase-1 Upgrade Parameters */
      "me11_phase1_tmbMpcBlockMe1a",
      "me11_phase1_tmbAlctTrigEnable",
      "me11_phase1_tmbClctTrigEnable",
      "me11_phase1_tmbMatchTrigEnable",
      "me11_phase1_tmbMatchTrigWindowSize",
      "me11_phase1_tmbTmbL1aWindowSize",
      "me11_phase1_tmbEarlyTbins",
      "me11_phase1_tmbCrossBxAlgorithm",
      "me11_phase1_maxME11LCTs",
  };

  COND_SERIALIZABLE;
};

#endif
