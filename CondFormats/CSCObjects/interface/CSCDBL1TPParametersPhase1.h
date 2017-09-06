#ifndef CSCObjects_CSCDBL1TPParametersPhase1_h
#define CSCObjects_CSCDBL1TPParametersPhase1_h

#include "CondFormats/Serialization/interface/Serializable.h"

/** \class CSCDBL1TPParametersPhase1
 *  \author Sven Dildick
 *
 * Description: Configuration parameters needed for the Phase-1 upgrade of the level-1
 *              CSC trigger primitives emulator. ME1/1 ALCT, CLCT and TMB have own members.
 *
 */

class CSCDBL1TPParametersPhase1
{
 public:
  CSCDBL1TPParametersPhase1();
  ~CSCDBL1TPParametersPhase1();

  /// Section 1: getters for CSC L1 TPG parameters ///

  /* Parameters for 2007 version of ALCT firmware */
  unsigned int alctFifoTbins() const;
  unsigned int alctFifoPretrig() const;
  unsigned int alctDriftDelay() const;
  unsigned int alctNplanesHitPretrig() const;
  unsigned int alctNplanesHitPattern() const;
  unsigned int alctNplanesHitAccelPretrig() const;
  unsigned int alctNplanesHitAccelPattern() const;
  unsigned int alctTrigMode() const;
  unsigned int alctAccelMode() const;
  unsigned int alctL1aWindowWidth() const;
  unsigned int alctEarlyTbins() const;

  /* ME11 ALCT Phase-1 Upgrade Parameters */
  unsigned int me11_phase1_alctFifoTbins() const;
  unsigned int me11_phase1_alctFifoPretrig() const;
  unsigned int me11_phase1_alctDriftDelay() const;
  unsigned int me11_phase1_alctNplanesHitPretrig() const;
  unsigned int me11_phase1_alctNplanesHitPattern() const;
  unsigned int me11_phase1_alctNplanesHitAccelPretrig() const;
  unsigned int me11_phase1_alctNplanesHitAccelPattern() const;
  unsigned int me11_phase1_alctTrigMode() const;
  unsigned int me11_phase1_alctAccelMode() const;
  unsigned int me11_phase1_alctL1aWindowWidth() const;
  unsigned int me11_phase1_alctEarlyTbins() const;
  bool         me11_phase1_alctNarrowMaskForR1a() const;
  unsigned int me11_phase1_alctHitPersist() const;
  unsigned int me11_phase1_alctGhostCancellationBxDepth() const;
  bool         me11_phase1_alctGhostCancellationSideQuality() const;
  unsigned int me11_phase1_alctPretrigDeadtime() const;
  bool         me11_phase1_alctUseCorrectedBx() const;

  /* Parameters for 2007 version of CLCT firmware */
  unsigned int clctFifoTbins() const;
  unsigned int clctFifoPretrig() const;
  unsigned int clctHitPersist() const;
  unsigned int clctDriftDelay() const;
  unsigned int clctNplanesHitPretrig() const;
  unsigned int clctNplanesHitPattern() const;
  unsigned int clctPidThreshPretrig() const;
  unsigned int clctMinSeparation() const;

  /* ME11 CLCT Phase-1 Upgrade Parameters */
  unsigned int me11_phase1_clctFifoTbins() const;
  unsigned int me11_phase1_clctFifoPretrig() const;
  unsigned int me11_phase1_clctHitPersist() const;
  unsigned int me11_phase1_clctDriftDelay() const;
  unsigned int me11_phase1_clctNplanesHitPretrig() const;
  unsigned int me11_phase1_clctNplanesHitPattern() const;
  unsigned int me11_phase1_clctPidThreshPretrig() const;
  unsigned int me11_phase1_clctMinSeparation() const;
  unsigned int me11_phase1_clctStartBxShift() const;
  bool         me11_phase1_useDeadTimeZoning() const;
  unsigned int me11_phase1_clctStateMachineZone() const;
  bool         me11_phase1_useDynamicStateMachineZone() const;
  unsigned int me11_phase1_clctPretriggerTriggerZone() const;
  bool         me11_phase1_clctUseCorrectedBx() const;

  /* Parameters for 2007 version of TMB firmware */
  unsigned int tmbMpcBlockMe1a() const;
  unsigned int tmbAlctTrigEnable() const;
  unsigned int tmbClctTrigEnable() const;
  unsigned int tmbMatchTrigEnable() const;
  unsigned int tmbMatchTrigWindowSize() const;
  unsigned int tmbTmbL1aWindowSize() const;

  /* ME11 TMB Phase-1 Upgrade Parameters */
  unsigned int me11_phase1_tmbMpcBlockMe1a() const;
  unsigned int me11_phase1_tmbAlctTrigEnable() const;
  unsigned int me11_phase1_tmbClctTrigEnable() const;
  unsigned int me11_phase1_tmbMatchTrigEnable() const;
  unsigned int me11_phase1_tmbMatchTrigWindowSize() const;
  unsigned int me11_phase1_tmbTmbL1aWindowSize() const;
  unsigned int me11_phase1_tmbEarlyTbins() const;
  bool         me11_phase1_tmbReadoutEarliest2() const;
  bool         me11_phase1_tmbDropUsedAlcts() const;
  bool         me11_phase1_clctToAlct() const;
  bool         me11_phase1_tmbDropUsedClcts() const;
  bool         me11_phase1_matchEarliestAlctME11Only() const;
  bool         me11_phase1_matchEarliestClctME11Only() const;
  unsigned int me11_phase1_tmbCrossBxAlgorithm() const;
  bool         me11_phase1_maxME11LCTs() const;

  /// Section 2: setters for CSC L1 TPG parameters ///

  /* Parameters for 2007 version of ALCT firmware */
  void setAlctFifoTbins(const unsigned int theValue);
  void setAlctFifoPretrig(const unsigned int theValue);
  void setAlctDriftDelay(const unsigned int theValue);
  void setAlctNplanesHitPretrig(const unsigned int theValue);
  void setAlctNplanesHitPattern(const unsigned int theValue);
  void setAlctNplanesHitAccelPretrig(const unsigned int theValue);
  void setAlctNplanesHitAccelPattern(const unsigned int theValue);
  void setAlctTrigMode(const unsigned int theValue);
  void setAlctAccelMode(const unsigned int theValue);
  void setAlctL1aWindowWidth(const unsigned int theValue);
  void setAlctEarlyTbins(const unsigned int theValue);

  /* ME11 ALCT Phase-1 Upgrade Parameters */
  void setAlctFifoTbins_me11_phase1(const unsigned int theValue);
  void setAlctFifoPretrig_me11_phase1(const unsigned int theValue);
  void setAlctDriftDelay_me11_phase1(const unsigned int theValue);
  void setAlctNplanesHitPretrig_me11_phase1(const unsigned int theValue);
  void setAlctNplanesHitPattern_me11_phase1(const unsigned int theValue);
  void setAlctNplanesHitAccelPretrig_me11_phase1(const unsigned int theValue);
  void setAlctNplanesHitAccelPattern_me11_phase1(const unsigned int theValue);
  void setAlctTrigMode_me11_phase1(const unsigned int theValue);
  void setAlctAccelMode_me11_phase1(const unsigned int theValue);
  void setAlctL1aWindowWidth_me11_phase1(const unsigned int theValue);
  void setAlctEarlyTbins_me11_phase1(const unsigned int theValue);
  void setAlctNarrowMaskForR1_me11_phase1(const bool theValue);
  void setAlctHitPersist_me11_phase1(const unsigned int theValue);
  void setAlctGhostCancellationBxDepth_me11_phase1(const unsigned int theValue);
  void setAlctGhostCancellationSideQuality_me11_phase1(const bool theValue);
  void setAlctPretrigDeadtime_me11_phase1(const unsigned int theValue);
  void setAlctUseCorrectedBx_me11_phase1(const bool theValue);

  /* Parameters for 2007 version of CLCT firmware */
  void setClctFifoTbins(const unsigned int theValue);
  void setClctFifoPretrig(const unsigned int theValue);
  void setClctHitPersist(const unsigned int theValue);
  void setClctDriftDelay(const unsigned int theValue);
  void setClctNplanesHitPretrig(const unsigned int theValue);
  void setClctNplanesHitPattern(const unsigned int theValue);
  void setClctPidThreshPretrig(const unsigned int theValue);
  void setClctMinSeparation(const unsigned int theValue);

  /* ME11 CLCT Phase-1 Upgrade Parameters */
  void setClctFifoTbins_me11_phase1(const unsigned int theValue);
  void setClctFifoPretrig_me11_phase1(const unsigned int theValue);
  void setClctHitPersist_me11_phase1(const unsigned int theValue);
  void setClctDriftDelay_me11_phase1(const unsigned int theValue);
  void setClctNplanesHitPretrig_me11_phase1(const unsigned int theValue);
  void setClctNplanesHitPattern_me11_phase1(const unsigned int theValue);
  void setClctPidThreshPretrig_me11_phase1(const unsigned int theValue);
  void setClctMinSeparation_me11_phase1(const unsigned int theValue);
  void setClctStartBxShift_me11_phase1(const unsigned int theValue);
  void setUseDeadTimeZoning_me11_phase1(const bool theValue);
  void setClctStateMachineZone_me11_phase1(const unsigned int theValue);
  void setUseDynamicStateMachineZone_me11_phase1(const bool theValue);
  void setClctPretriggerTriggerZone_me11_phase1(const unsigned int theValue);
  void setClctUseCorrectedBx_me11_phase1(const bool theValue);

  /* Parameters for 2007 version of TMB firmware */
  void setTmbMpcBlockMe1a(const unsigned int theValue);
  void setTmbAlctTrigEnable(const unsigned int theValue);
  void setTmbClctTrigEnable(const unsigned int theValue);
  void setTmbMatchTrigEnable(const unsigned int theValue);
  void setTmbMatchTrigWindowSize(const unsigned int theValue);
  void setTmbTmbL1aWindowSize(const unsigned int theValue);

  /* ME11 TMB Phase-1 Upgrade Parameters */
  void setTmbMpcBlockMe1a_me11_phase1(const unsigned int theValue);
  void setTmbAlctTrigEnable_me11_phase1(const unsigned int theValue);
  void setTmbClctTrigEnable_me11_phase1(const unsigned int theValue);
  void setTmbMatchTrigEnable_me11_phase1(const unsigned int theValue);
  void setTmbMatchTrigWindowSize_me11_phase1(const unsigned int theValue);
  void setTmbTmbL1aWindowSize_me11_phase1(const unsigned int theValue);
  void setTmbEarlyTbins_me11_phase1(const bool theValue);
  void setTmbReadoutEarliest2_me11_phase1(const bool theValue);
  void setTmbDropUsedAlcts_me11_phase1(const bool theValue);
  void setClctToAlct_me11_phase1(const bool theValue);
  void setTmbDropUsedClcts_me11_phase1(const bool theValue);
  void setMatchEarliestAlctME11Only_me11_phase1(const bool theValue);
  void setMatchEarliestClctME11Only_me11_phase1(const bool theValue);
  void setTmbCrossBxAlgorithm_me11_phase1(const unsigned int theValue);
  void setMaxME11LCTs_me11_phase1(const unsigned int theValue);

 private:
  /** ALCT configuration parameters. */
  unsigned int m_alct_fifo_tbins, m_alct_fifo_pretrig;
  unsigned int m_alct_drift_delay;
  unsigned int m_alct_nplanes_hit_pretrig, m_alct_nplanes_hit_accel_pretrig;
  unsigned int m_alct_nplanes_hit_pattern, m_alct_nplanes_hit_accel_pattern;
  unsigned int m_alct_trig_mode, m_alct_accel_mode, m_alct_l1a_window_width;
  unsigned int m_alct_early_time_bins;

  /* ALCT Phase-1 upgrade configuration parameters */
  unsigned int m_me11_phase1_alct_fifo_tbins, m_me11_phase1_alct_fifo_pretrig;
  unsigned int m_me11_phase1_alct_drift_delay;
  unsigned int m_me11_phase1_alct_nplanes_hit_pretrig, m_me11_phase1_alct_nplanes_hit_accel_pretrig;
  unsigned int m_me11_phase1_alct_nplanes_hit_pattern, m_me11_phase1_alct_nplanes_hit_accel_pattern;
  unsigned int m_me11_phase1_alct_trig_mode, m_me11_phase1_alct_accel_mode, m_me11_phase1_alct_l1a_window_width;
  unsigned int m_me11_phase1_alct_early_time_bins, m_me11_phase1_alct_hit_persist;
  bool m_me11_phase1_alct_narrow_maks_for_r1a;
  unsigned int m_me11_phase1_alct_ghost_cancellation_bx_depth, m_me11_phase1_alct_ghost_cancellation_side_quality;
  unsigned int m_me11_phase1_alct_pretrig_dead_time;
  bool m_me11_phase1_alct_use_corrected_bx;

  /** CLCT configuration parameters. */
  unsigned int m_clct_fifo_tbins,  m_clct_fifo_pretrig;
  unsigned int m_clct_hit_persist, m_clct_drift_delay;
  unsigned int m_clct_nplanes_hit_pretrig, m_clct_nplanes_hit_pattern;
  unsigned int m_clct_pid_thresh_pretrig;
  unsigned int m_clct_min_separation;

  /* CLCT Phase-1 upgrade configuration parameters */
  unsigned int m_me11_phase1_clct_fifo_tbins,  m_me11_phase1_clct_fifo_pretrig;
  unsigned int m_me11_phase1_clct_hit_persist, m_me11_phase1_clct_drift_delay;
  unsigned int m_me11_phase1_clct_nplanes_hit_pretrig, m_me11_phase1_clct_nplanes_hit_pattern;
  unsigned int m_me11_phase1_clct_pid_thresh_pretrig;
  unsigned int m_me11_phase1_clct_min_separation;
  unsigned int m_me11_phase1_clct_start_bx_shift;
  bool m_me11_phase1_clct_use_deadtime_zoning;
  unsigned int m_me11_phase1_clct_state_machine_zone;
  bool m_me11_phase1_clct_use_dynamic_state_matchine_zone;
  unsigned int m_me11_phase1_clct_pretrigger_trigger_zone;
  bool m_me11_phase1_clct_use_corrected_bx;

  /** TMB configuration parameters. */
  unsigned int m_mpc_block_me1a;
  unsigned int m_alct_trig_enable, m_clct_trig_enable;
  unsigned int m_match_trig_enable;
  unsigned int m_match_trig_window_size, m_tmb_l1a_window_size;

  /* ME11 TMB Phase-1 upgrade configuration parameters */
  unsigned int m_me11_phase1_mpc_block_me1a;
  unsigned int m_me11_phase1_alct_trig_enable, m_me11_phase1_clct_trig_enable;
  unsigned int m_me11_phase1_match_trig_enable;
  unsigned int m_me11_phase1_match_trig_window_size, m_me11_phase1_tmb_l1a_window_size;
  unsigned int m_me11_phase1_early_time_bins;
  bool m_me11_phase1_readout_earliest2;
  bool m_me11_phase1_clct_to_alct, m_me11_phase1_drop_used_clcts, m_me11_phase1_drop_used_alcts;
  bool m_me11_phase1_match_earliest_alct_me11_only;
  bool m_me11_phase1_match_earliest_clct_me11_only;
  unsigned int m_me11_phase1_cross_bx_algorithm, m_me11_phase1_max_me11_lcts;

 COND_SERIALIZABLE;
};

#endif
