#include "CondFormats/CSCObjects/interface/CSCDBL1TPParametersPhase1.h"

//----------------
// Constructors --
//----------------

CSCDBL1TPParametersPhase1::CSCDBL1TPParametersPhase1() {
}

//----------------
//  Destructor  --
//----------------

CSCDBL1TPParametersPhase1::~CSCDBL1TPParametersPhase1() {
}

  /* Parameters for 2007 version of ALCT firmware */
unsigned int CSCDBL1TPParametersPhase1::alctFifoTbins() const
{
  return m_alct_fifo_tbins;
}

unsigned int CSCDBL1TPParametersPhase1::alctFifoPretrig() const
{
  return m_alct_fifo_pretrig;
}

unsigned int CSCDBL1TPParametersPhase1::alctDriftDelay() const
{
  return m_alct_drift_delay;
}

unsigned int CSCDBL1TPParametersPhase1::alctNplanesHitPretrig() const
{
  return m_alct_nplanes_hit_pretrig;
}

unsigned int CSCDBL1TPParametersPhase1::alctNplanesHitPattern() const
{
  return m_alct_nplanes_hit_pattern;
}

unsigned int CSCDBL1TPParametersPhase1::alctNplanesHitAccelPretrig() const
{
  return m_alct_nplanes_hit_accel_pretrig;
}

unsigned int CSCDBL1TPParametersPhase1::alctNplanesHitAccelPattern() const
{
  return m_alct_nplanes_hit_accel_pattern;
}

unsigned int CSCDBL1TPParametersPhase1::alctTrigMode() const
{
  return m_alct_trig_mode;
}

unsigned int CSCDBL1TPParametersPhase1::alctAccelMode() const
{
  return m_alct_accel_mode;
}

unsigned int CSCDBL1TPParametersPhase1::alctL1aWindowWidth() const
{
  return m_alct_l1a_window_width;
}

unsigned int CSCDBL1TPParametersPhase1::alctEarlyTbins() const
{
  return m_alct_early_time_bins;
}


  /* ME11 ALCT Phase-1 Upgrade Parameters */
unsigned int CSCDBL1TPParametersPhase1::me11_phase1_alctFifoTbins() const
{
  return m_me11_phase1_alct_fifo_tbins;
}

unsigned int CSCDBL1TPParametersPhase1::me11_phase1_alctFifoPretrig() const
{
  return m_me11_phase1_alct_fifo_pretrig;
}

unsigned int CSCDBL1TPParametersPhase1::me11_phase1_alctDriftDelay() const
{
  return m_me11_phase1_alct_drift_delay;
}

unsigned int CSCDBL1TPParametersPhase1::me11_phase1_alctNplanesHitPretrig() const
{
  return m_me11_phase1_alct_nplanes_hit_pretrig;
}

unsigned int CSCDBL1TPParametersPhase1::me11_phase1_alctNplanesHitPattern() const
{
  return m_me11_phase1_alct_nplanes_hit_pattern;
}

unsigned int CSCDBL1TPParametersPhase1::me11_phase1_alctNplanesHitAccelPretrig() const
{
  return m_me11_phase1_alct_nplanes_hit_accel_pretrig;
}

unsigned int CSCDBL1TPParametersPhase1::me11_phase1_alctNplanesHitAccelPattern() const
{
  return m_me11_phase1_alct_nplanes_hit_accel_pattern;
}

unsigned int CSCDBL1TPParametersPhase1::me11_phase1_alctTrigMode() const
{
  return m_me11_phase1_alct_trig_mode;
}

unsigned int CSCDBL1TPParametersPhase1::me11_phase1_alctAccelMode() const
{
  return m_me11_phase1_alct_accel_mode;
}

unsigned int CSCDBL1TPParametersPhase1::me11_phase1_alctL1aWindowWidth() const
{
  return m_me11_phase1_alct_l1a_window_width;
}

unsigned int CSCDBL1TPParametersPhase1::me11_phase1_alctEarlyTbins() const
{
  return m_me11_phase1_alct_early_time_bins;
}

bool CSCDBL1TPParametersPhase1::me11_phase1_alctNarrowMaskForR1a() const
{
  return m_me11_phase1_alct_narrow_maks_for_r1a;
}

unsigned int CSCDBL1TPParametersPhase1::me11_phase1_alctHitPersist() const
{
  return m_me11_phase1_alct_hit_persist;
}

unsigned int CSCDBL1TPParametersPhase1::me11_phase1_alctGhostCancellationBxDepth() const
{
  return m_me11_phase1_alct_ghost_cancellation_bx_depth;
}

bool CSCDBL1TPParametersPhase1::me11_phase1_alctGhostCancellationSideQuality() const
{
  return m_me11_phase1_alct_ghost_cancellation_side_quality;
}

unsigned int CSCDBL1TPParametersPhase1::me11_phase1_alctPretrigDeadtime() const
{
  return m_me11_phase1_alct_pretrig_dead_time;
}

bool CSCDBL1TPParametersPhase1::me11_phase1_alctUseCorrectedBx() const
{
  return m_me11_phase1_alct_use_corrected_bx;
}


  /* Parameters for 2007 version of CLCT firmware */
unsigned int CSCDBL1TPParametersPhase1::clctFifoTbins() const
{
  return m_clct_fifo_tbins;
}

unsigned int CSCDBL1TPParametersPhase1::clctFifoPretrig() const
{
  return m_clct_fifo_pretrig;
}

unsigned int CSCDBL1TPParametersPhase1::clctHitPersist() const
{
  return m_clct_hit_persist;
}

unsigned int CSCDBL1TPParametersPhase1::clctDriftDelay() const
{
  return m_clct_drift_delay;
}

unsigned int CSCDBL1TPParametersPhase1::clctNplanesHitPretrig() const
{
  return m_clct_nplanes_hit_pretrig;
}

unsigned int CSCDBL1TPParametersPhase1::clctNplanesHitPattern() const
{
  return m_clct_nplanes_hit_pattern;
}

unsigned int CSCDBL1TPParametersPhase1::clctPidThreshPretrig() const
{
  return m_clct_pid_thresh_pretrig;
}

unsigned int CSCDBL1TPParametersPhase1::clctMinSeparation() const
{
  return m_clct_min_separation;
}


  /* ME11 CLCT Phase-1 Upgrade Parameters */
unsigned int CSCDBL1TPParametersPhase1::me11_phase1_clctFifoTbins() const
{
  return m_me11_phase1_clct_fifo_tbins;
}

unsigned int CSCDBL1TPParametersPhase1::me11_phase1_clctFifoPretrig() const
{
  return m_me11_phase1_clct_fifo_pretrig;
}

unsigned int CSCDBL1TPParametersPhase1::me11_phase1_clctHitPersist() const
{
  return m_me11_phase1_clct_hit_persist;
}

unsigned int CSCDBL1TPParametersPhase1::me11_phase1_clctDriftDelay() const
{
  return m_me11_phase1_clct_drift_delay;
}

unsigned int CSCDBL1TPParametersPhase1::me11_phase1_clctNplanesHitPretrig() const
{
  return m_me11_phase1_clct_nplanes_hit_pretrig;
}

unsigned int CSCDBL1TPParametersPhase1::me11_phase1_clctNplanesHitPattern() const
{
  return m_me11_phase1_clct_nplanes_hit_pattern;
}

unsigned int CSCDBL1TPParametersPhase1::me11_phase1_clctPidThreshPretrig() const
{
  return m_me11_phase1_clct_pid_thresh_pretrig;
}

unsigned int CSCDBL1TPParametersPhase1::me11_phase1_clctMinSeparation() const
{
  return m_me11_phase1_clct_min_separation;
}

unsigned int CSCDBL1TPParametersPhase1::me11_phase1_clctStartBxShift() const
{
  return m_me11_phase1_clct_start_bx_shift;
}

bool CSCDBL1TPParametersPhase1::me11_phase1_useDeadTimeZoning() const
{
  return m_me11_phase1_clct_use_deadtime_zoning;
}

unsigned int CSCDBL1TPParametersPhase1::me11_phase1_clctStateMachineZone() const
{
  return m_me11_phase1_clct_state_machine_zone;
}

bool CSCDBL1TPParametersPhase1::me11_phase1_useDynamicStateMachineZone() const
{
  return m_me11_phase1_clct_use_dynamic_state_matchine_zone;
}

unsigned int CSCDBL1TPParametersPhase1::me11_phase1_clctPretriggerTriggerZone() const
{
  return m_me11_phase1_clct_pretrigger_trigger_zone;
}

bool CSCDBL1TPParametersPhase1::me11_phase1_clctUseCorrectedBx() const
{
  return m_me11_phase1_clct_use_corrected_bx;
}


  /* Parameters for 2007 version of TMB firmware */
unsigned int CSCDBL1TPParametersPhase1::tmbMpcBlockMe1a() const
{
  return m_mpc_block_me1a;
}

unsigned int CSCDBL1TPParametersPhase1::tmbAlctTrigEnable() const
{
  return m_alct_trig_enable;
}

unsigned int CSCDBL1TPParametersPhase1::tmbClctTrigEnable() const
{
  return m_clct_trig_enable;
}

unsigned int CSCDBL1TPParametersPhase1::tmbMatchTrigEnable() const
{
  return m_match_trig_enable;
}

unsigned int CSCDBL1TPParametersPhase1::tmbMatchTrigWindowSize() const
{
  return m_match_trig_window_size;
}

unsigned int CSCDBL1TPParametersPhase1::tmbTmbL1aWindowSize() const
{
  return m_tmb_l1a_window_size;
}


  /* ME11 TMB Phase-1 Upgrade Parameters */
unsigned int CSCDBL1TPParametersPhase1::me11_phase1_tmbMpcBlockMe1a() const
{
  return m_me11_phase1_mpc_block_me1a;
}

unsigned int CSCDBL1TPParametersPhase1::me11_phase1_tmbAlctTrigEnable() const
{
  return m_me11_phase1_alct_trig_enable;
}

unsigned int CSCDBL1TPParametersPhase1::me11_phase1_tmbClctTrigEnable() const
{
  return m_me11_phase1_clct_trig_enable;
}

unsigned int CSCDBL1TPParametersPhase1::me11_phase1_tmbMatchTrigEnable() const
{
  return m_me11_phase1_match_trig_enable;
}

unsigned int CSCDBL1TPParametersPhase1::me11_phase1_tmbMatchTrigWindowSize() const
{
  return m_me11_phase1_match_trig_window_size;
}

unsigned int CSCDBL1TPParametersPhase1::me11_phase1_tmbTmbL1aWindowSize() const
{
  return m_me11_phase1_tmb_l1a_window_size;
}

unsigned int CSCDBL1TPParametersPhase1::me11_phase1_tmbEarlyTbins() const
{
  return m_me11_phase1_early_time_bins;
}

bool CSCDBL1TPParametersPhase1::me11_phase1_tmbReadoutEarliest2() const
{
  return m_me11_phase1_readout_earliest2;
}

bool CSCDBL1TPParametersPhase1::me11_phase1_tmbDropUsedAlcts() const
{
  return m_me11_phase1_drop_used_alcts;
}

bool CSCDBL1TPParametersPhase1::me11_phase1_clctToAlct() const
{
  return m_me11_phase1_clct_to_alct;
}

bool CSCDBL1TPParametersPhase1::me11_phase1_tmbDropUsedClcts() const
{
  return m_me11_phase1_drop_used_clcts;
}

bool CSCDBL1TPParametersPhase1::me11_phase1_matchEarliestAlctME11Only() const
{
  return m_me11_phase1_match_earliest_alct_me11_only;
}

bool CSCDBL1TPParametersPhase1::me11_phase1_matchEarliestClctME11Only() const
{
  return m_me11_phase1_match_earliest_clct_me11_only;
}

unsigned int CSCDBL1TPParametersPhase1::me11_phase1_tmbCrossBxAlgorithm() const
{
  return m_me11_phase1_cross_bx_algorithm;
}

bool CSCDBL1TPParametersPhase1::me11_phase1_maxME11LCTs() const
{
  return m_me11_phase1_max_me11_lcts;
}


  /// Section 2: setters for CSC L1 TPG parameters ///

  /* Parameters for 2007 version of ALCT firmware */
void CSCDBL1TPParametersPhase1::setAlctFifoTbins(const unsigned int theValue)
{
  m_alct_fifo_tbins = theValue;
}

void CSCDBL1TPParametersPhase1::setAlctFifoPretrig(const unsigned int theValue)
{
  m_alct_fifo_pretrig = theValue;
}

void CSCDBL1TPParametersPhase1::setAlctDriftDelay(const unsigned int theValue)
{
  m_alct_drift_delay = theValue;
}

void CSCDBL1TPParametersPhase1::setAlctNplanesHitPretrig(const unsigned int theValue)
{
  m_alct_nplanes_hit_pretrig = theValue;
}

void CSCDBL1TPParametersPhase1::setAlctNplanesHitPattern(const unsigned int theValue)
{
  m_alct_nplanes_hit_pattern = theValue;
}

void CSCDBL1TPParametersPhase1::setAlctNplanesHitAccelPretrig(const unsigned int theValue)
{
  m_alct_nplanes_hit_accel_pretrig = theValue;
}

void CSCDBL1TPParametersPhase1::setAlctNplanesHitAccelPattern(const unsigned int theValue)
{
  m_alct_nplanes_hit_accel_pattern = theValue;
}

void CSCDBL1TPParametersPhase1::setAlctTrigMode(const unsigned int theValue)
{
  m_alct_trig_mode = theValue;
}

void CSCDBL1TPParametersPhase1::setAlctAccelMode(const unsigned int theValue)
{
  m_alct_accel_mode = theValue;
}

void CSCDBL1TPParametersPhase1::setAlctL1aWindowWidth(const unsigned int theValue)
{
  m_alct_l1a_window_width = theValue;
}

void CSCDBL1TPParametersPhase1::setAlctEarlyTbins(const unsigned int theValue)
{
  m_alct_early_time_bins = theValue;
}

  /* ME11 ALCT Phase-1 Upgrade Parameters */
void CSCDBL1TPParametersPhase1::setAlctFifoTbins_me11_phase1(const unsigned int theValue)
{
  m_me11_phase1_alct_fifo_tbins = theValue;
}

void CSCDBL1TPParametersPhase1::setAlctFifoPretrig_me11_phase1(const unsigned int theValue)
{
  m_me11_phase1_alct_fifo_pretrig = theValue;
}

void CSCDBL1TPParametersPhase1::setAlctDriftDelay_me11_phase1(const unsigned int theValue)
{
  m_me11_phase1_alct_drift_delay = theValue;
}

void CSCDBL1TPParametersPhase1::setAlctNplanesHitPretrig_me11_phase1(const unsigned int theValue)
{
  m_me11_phase1_alct_nplanes_hit_pretrig = theValue;
}

void CSCDBL1TPParametersPhase1::setAlctNplanesHitPattern_me11_phase1(const unsigned int theValue)
{
  m_me11_phase1_alct_nplanes_hit_pattern = theValue;
}

void CSCDBL1TPParametersPhase1::setAlctNplanesHitAccelPretrig_me11_phase1(const unsigned int theValue)
{
  m_me11_phase1_alct_nplanes_hit_accel_pretrig = theValue;
}

void CSCDBL1TPParametersPhase1::setAlctNplanesHitAccelPattern_me11_phase1(const unsigned int theValue)
{
  m_me11_phase1_alct_nplanes_hit_accel_pattern = theValue;
}

void CSCDBL1TPParametersPhase1::setAlctTrigMode_me11_phase1(const unsigned int theValue)
{
  m_me11_phase1_alct_trig_mode = theValue;
}

void CSCDBL1TPParametersPhase1::setAlctAccelMode_me11_phase1(const unsigned int theValue)
{
  m_me11_phase1_alct_accel_mode = theValue;
}

void CSCDBL1TPParametersPhase1::setAlctL1aWindowWidth_me11_phase1(const unsigned int theValue)
{
  m_me11_phase1_alct_l1a_window_width = theValue;
}

void CSCDBL1TPParametersPhase1::setAlctEarlyTbins_me11_phase1(const unsigned int theValue)
{
  m_me11_phase1_alct_early_time_bins = theValue;
}

void CSCDBL1TPParametersPhase1::setAlctNarrowMaskForR1_me11_phase1(const bool theValue)
{
  m_me11_phase1_alct_narrow_maks_for_r1a = theValue;
}

void CSCDBL1TPParametersPhase1::setAlctHitPersist_me11_phase1(const unsigned int theValue)
{
  m_me11_phase1_alct_hit_persist = theValue;
}

void CSCDBL1TPParametersPhase1::setAlctGhostCancellationBxDepth_me11_phase1(const unsigned int theValue)
{
  m_me11_phase1_alct_ghost_cancellation_bx_depth = theValue;
}

void CSCDBL1TPParametersPhase1::setAlctGhostCancellationSideQuality_me11_phase1(const bool theValue)
{
  m_me11_phase1_alct_ghost_cancellation_side_quality = theValue;
}

void CSCDBL1TPParametersPhase1::setAlctPretrigDeadtime_me11_phase1(const unsigned int theValue)
{
  m_me11_phase1_alct_pretrig_dead_time = theValue;
}

void CSCDBL1TPParametersPhase1::setAlctUseCorrectedBx_me11_phase1(const bool theValue)
{
  m_me11_phase1_alct_use_corrected_bx = theValue;
}


  /* Parameters for 2007 version of CLCT firmware */
void CSCDBL1TPParametersPhase1::setClctFifoTbins(const unsigned int theValue)
{
  m_clct_fifo_tbins = theValue;
}

void CSCDBL1TPParametersPhase1::setClctFifoPretrig(const unsigned int theValue)
{
  m_clct_fifo_pretrig = theValue;
}

void CSCDBL1TPParametersPhase1::setClctHitPersist(const unsigned int theValue)
{
  m_clct_hit_persist = theValue;
}

void CSCDBL1TPParametersPhase1::setClctDriftDelay(const unsigned int theValue)
{
  m_clct_drift_delay = theValue;
}

void CSCDBL1TPParametersPhase1::setClctNplanesHitPretrig(const unsigned int theValue)
{
  m_clct_nplanes_hit_pretrig = theValue;
}

void CSCDBL1TPParametersPhase1::setClctNplanesHitPattern(const unsigned int theValue)
{
  m_clct_nplanes_hit_pattern = theValue;
}

void CSCDBL1TPParametersPhase1::setClctPidThreshPretrig(const unsigned int theValue)
{
  m_clct_pid_thresh_pretrig = theValue;
}

void CSCDBL1TPParametersPhase1::setClctMinSeparation(const unsigned int theValue)
{
  m_clct_min_separation = theValue;
}


  /* ME11 CLCT Phase-1 Upgrade Parameters */
void CSCDBL1TPParametersPhase1::setClctFifoTbins_me11_phase1(const unsigned int theValue)
{
  m_me11_phase1_clct_fifo_tbins = theValue;
}

void CSCDBL1TPParametersPhase1::setClctFifoPretrig_me11_phase1(const unsigned int theValue)
{
  m_me11_phase1_clct_fifo_pretrig = theValue;
}

void CSCDBL1TPParametersPhase1::setClctHitPersist_me11_phase1(const unsigned int theValue)
{
  m_me11_phase1_clct_hit_persist = theValue;
}

void CSCDBL1TPParametersPhase1::setClctDriftDelay_me11_phase1(const unsigned int theValue)
{
  m_me11_phase1_clct_drift_delay = theValue;
}

void CSCDBL1TPParametersPhase1::setClctNplanesHitPretrig_me11_phase1(const unsigned int theValue)
{
  m_me11_phase1_clct_nplanes_hit_pretrig = theValue;
}

void CSCDBL1TPParametersPhase1::setClctNplanesHitPattern_me11_phase1(const unsigned int theValue)
{
  m_me11_phase1_clct_nplanes_hit_pattern = theValue;
}

void CSCDBL1TPParametersPhase1::setClctPidThreshPretrig_me11_phase1(const unsigned int theValue)
{
  m_me11_phase1_clct_pid_thresh_pretrig = theValue;
}

void CSCDBL1TPParametersPhase1::setClctMinSeparation_me11_phase1(const unsigned int theValue)
{
  m_me11_phase1_clct_min_separation = theValue;
}

void CSCDBL1TPParametersPhase1::setClctStartBxShift_me11_phase1(const unsigned int theValue)
{
  m_me11_phase1_clct_start_bx_shift = theValue;
}

void CSCDBL1TPParametersPhase1::setUseDeadTimeZoning_me11_phase1(const bool theValue)
{
  m_me11_phase1_clct_use_deadtime_zoning = theValue;
}

void CSCDBL1TPParametersPhase1::setClctStateMachineZone_me11_phase1(const unsigned int theValue)
{
  m_me11_phase1_clct_state_machine_zone = theValue;
}

void CSCDBL1TPParametersPhase1::setUseDynamicStateMachineZone_me11_phase1(const bool theValue)
{
  m_me11_phase1_clct_use_dynamic_state_matchine_zone = theValue;
}

void CSCDBL1TPParametersPhase1::setClctPretriggerTriggerZone_me11_phase1(const unsigned int theValue)
{
  m_me11_phase1_clct_pretrigger_trigger_zone = theValue;
}

void CSCDBL1TPParametersPhase1::setClctUseCorrectedBx_me11_phase1(const bool theValue)
{
  m_me11_phase1_clct_use_corrected_bx = theValue;
}


  /* Parameters for 2007 version of TMB firmware */
void CSCDBL1TPParametersPhase1::setTmbMpcBlockMe1a(const unsigned int theValue)
{
  m_mpc_block_me1a = theValue;
}

void CSCDBL1TPParametersPhase1::setTmbAlctTrigEnable(const unsigned int theValue)
{
  m_alct_trig_enable = theValue;
}

void CSCDBL1TPParametersPhase1::setTmbClctTrigEnable(const unsigned int theValue)
{
  m_clct_trig_enable = theValue;
}

void CSCDBL1TPParametersPhase1::setTmbMatchTrigEnable(const unsigned int theValue)
{
  m_match_trig_enable = theValue;
}

void CSCDBL1TPParametersPhase1::setTmbMatchTrigWindowSize(const unsigned int theValue)
{
  m_match_trig_window_size = theValue;
}

void CSCDBL1TPParametersPhase1::setTmbTmbL1aWindowSize(const unsigned int theValue)
{
  m_tmb_l1a_window_size = theValue;
}


  /* ME11 TMB Phase-1 Upgrade Parameters */
void CSCDBL1TPParametersPhase1::setTmbMpcBlockMe1a_me11_phase1(const unsigned int theValue)
{
  m_me11_phase1_mpc_block_me1a = theValue;
}

void CSCDBL1TPParametersPhase1::setTmbAlctTrigEnable_me11_phase1(const unsigned int theValue)
{
  m_me11_phase1_alct_trig_enable = theValue;
}

void CSCDBL1TPParametersPhase1::setTmbClctTrigEnable_me11_phase1(const unsigned int theValue)
{
  m_me11_phase1_clct_trig_enable = theValue;
}

void CSCDBL1TPParametersPhase1::setTmbMatchTrigEnable_me11_phase1(const unsigned int theValue)
{
  m_me11_phase1_match_trig_enable = theValue;
}

void CSCDBL1TPParametersPhase1::setTmbMatchTrigWindowSize_me11_phase1(const unsigned int theValue)
{
  m_me11_phase1_match_trig_window_size = theValue;
}

void CSCDBL1TPParametersPhase1::setTmbTmbL1aWindowSize_me11_phase1(const unsigned int theValue)
{
  m_me11_phase1_tmb_l1a_window_size = theValue;
}

void CSCDBL1TPParametersPhase1::setTmbEarlyTbins_me11_phase1(const bool theValue)
{
  m_me11_phase1_early_time_bins = theValue;
}

void CSCDBL1TPParametersPhase1::setTmbReadoutEarliest2_me11_phase1(const bool theValue)
{
 m_me11_phase1_readout_earliest2 = theValue;
}

void CSCDBL1TPParametersPhase1::setTmbDropUsedAlcts_me11_phase1(const bool theValue)
{
  m_me11_phase1_drop_used_alcts = theValue;
}

void CSCDBL1TPParametersPhase1::setClctToAlct_me11_phase1(const bool theValue)
{
 m_me11_phase1_clct_to_alct = theValue;
}

void CSCDBL1TPParametersPhase1::setTmbDropUsedClcts_me11_phase1(const bool theValue)
{
  m_me11_phase1_drop_used_clcts = theValue;
}

void CSCDBL1TPParametersPhase1::setMatchEarliestAlctME11Only_me11_phase1(const bool theValue)
{
  m_me11_phase1_match_earliest_alct_me11_only = theValue;
}

void CSCDBL1TPParametersPhase1::setMatchEarliestClctME11Only_me11_phase1(const bool theValue)
{
  m_me11_phase1_match_earliest_clct_me11_only = theValue;
}

void CSCDBL1TPParametersPhase1::setTmbCrossBxAlgorithm_me11_phase1(const unsigned int theValue)
{
  m_me11_phase1_match_earliest_clct_me11_only = theValue;
}

void CSCDBL1TPParametersPhase1::setMaxME11LCTs_me11_phase1(const unsigned int theValue)
{
  m_me11_phase1_max_me11_lcts = theValue;
}

