#ifndef CSCObjects_CSCL1TPParameters_h
#define CSCObjects_CSCL1TPParameters_h

#include "CondFormats/Serialization/interface/Serializable.h"

/** \class CSCL1TPParameters
 *  \author Slava Valuev
 *
 * Description: Configuration parameters needed for the Level-1 CSC Trigger
 *              Primitives emulator.  Expected to be stored in and retrieved
 *              from the conditions database.
 */

class CSCL1TPParameters {
public:
  CSCL1TPParameters();
  ~CSCL1TPParameters();

  /** returns ALCT fifo_tbins */
  inline unsigned int alctFifoTbins() const { return m_alct_fifo_tbins; }

  /** returns ALCT fifo_pretrig */
  inline unsigned int alctFifoPretrig() const { return m_alct_fifo_pretrig; }

  /** returns ALCT drift_delay */
  inline unsigned int alctDriftDelay() const { return m_alct_drift_delay; }

  /** returns ALCT nplanes_hit_pretrig */
  inline unsigned int alctNplanesHitPretrig() const { return m_alct_nplanes_hit_pretrig; }

  /** returns ALCT nplanes_hit_pattern */
  inline unsigned int alctNplanesHitPattern() const { return m_alct_nplanes_hit_pattern; }

  /** returns ALCT nplanes_hit_accel_pretrig */
  inline unsigned int alctNplanesHitAccelPretrig() const { return m_alct_nplanes_hit_accel_pretrig; }

  /** returns ALCT nplanes_hit_accel_pattern */
  inline unsigned int alctNplanesHitAccelPattern() const { return m_alct_nplanes_hit_accel_pattern; }

  /** returns ALCT trig_mode */
  inline unsigned int alctTrigMode() const { return m_alct_trig_mode; }

  /** returns ALCT accel_mode */
  inline unsigned int alctAccelMode() const { return m_alct_accel_mode; }

  /** returns ALCT l1a_window_width */
  inline unsigned int alctL1aWindowWidth() const { return m_alct_l1a_window_width; }

  /** returns CLCT fifo_tbins */
  inline unsigned int clctFifoTbins() const { return m_clct_fifo_tbins; }

  /** returns CLCT fifo_pretrig */
  inline unsigned int clctFifoPretrig() const { return m_clct_fifo_pretrig; }

  /** returns CLCT hit_persist */
  inline unsigned int clctHitPersist() const { return m_clct_hit_persist; }

  /** returns CLCT drift_delay */
  inline unsigned int clctDriftDelay() const { return m_clct_drift_delay; }

  /** returns CLCT nplanes_hit_pretrig */
  inline unsigned int clctNplanesHitPretrig() const { return m_clct_nplanes_hit_pretrig; }

  /** returns CLCT nplanes_hit_pattern */
  inline unsigned int clctNplanesHitPattern() const { return m_clct_nplanes_hit_pattern; }

  /** returns CLCT pid_thresh_pretrig */
  inline unsigned int clctPidThreshPretrig() const { return m_clct_pid_thresh_pretrig; }

  /** returns CLCT min_separation */
  inline unsigned int clctMinSeparation() const { return m_clct_min_separation; }

  /** sets ALCT fifo_tbins */
  void setAlctFifoTbins(const unsigned int theValue) { m_alct_fifo_tbins = theValue; }

  /** sets ALCT fifo_pretrig */
  void setAlctFifoPretrig(const unsigned int theValue) { m_alct_fifo_pretrig = theValue; }

  /** sets ALCT drift_delay */
  void setAlctDriftDelay(const unsigned int theValue) { m_alct_drift_delay = theValue; }

  /** sets ALCT nplanes_hit_pretrig */
  void setAlctNplanesHitPretrig(const unsigned int theValue) { m_alct_nplanes_hit_pretrig = theValue; }

  /** sets ALCT nplanes_hit_pattern */
  void setAlctNplanesHitPattern(const unsigned int theValue) { m_alct_nplanes_hit_pattern = theValue; }

  /** sets ALCT nplanes_hit_accel_pretrig */
  void setAlctNplanesHitAccelPretrig(const unsigned int theValue) { m_alct_nplanes_hit_accel_pretrig = theValue; }

  /** sets ALCT nplanes_hit_accel_pattern */
  void setAlctNplanesHitAccelPattern(const unsigned int theValue) { m_alct_nplanes_hit_accel_pattern = theValue; }

  /** sets ALCT trig_mode */
  void setAlctTrigMode(const unsigned int theValue) { m_alct_trig_mode = theValue; }

  /** sets ALCT accel_mode */
  void setAlctAccelMode(const unsigned int theValue) { m_alct_accel_mode = theValue; }

  /** sets ALCT l1a_window_width */
  void setAlctL1aWindowWidth(const unsigned int theValue) { m_alct_l1a_window_width = theValue; }

  /** sets CLCT fifo_tbins */
  void setClctFifoTbins(const unsigned int theValue) { m_clct_fifo_tbins = theValue; }

  /** sets CLCT fifo_pretrig */
  void setClctFifoPretrig(const unsigned int theValue) { m_clct_fifo_pretrig = theValue; }

  /** sets CLCT hit_persist */
  void setClctHitPersist(const unsigned int theValue) { m_clct_hit_persist = theValue; }

  /** sets CLCT drift_delay */
  void setClctDriftDelay(const unsigned int theValue) { m_clct_drift_delay = theValue; }

  /** sets CLCT nplanes_hit_pretrig */
  void setClctNplanesHitPretrig(const unsigned int theValue) { m_clct_nplanes_hit_pretrig = theValue; }

  /** sets CLCT nplanes_hit_pattern */
  void setClctNplanesHitPattern(const unsigned int theValue) { m_clct_nplanes_hit_pattern = theValue; }

  /** sets CLCT pid_thresh_pretrig */
  void setClctPidThreshPretrig(const unsigned int theValue) { m_clct_pid_thresh_pretrig = theValue; }

  /** sets CLCT min_separation */
  void setClctMinSeparation(const unsigned int theValue) { m_clct_min_separation = theValue; }

private:
  /** ALCT configuration parameters. */
  unsigned int m_alct_fifo_tbins, m_alct_fifo_pretrig;
  unsigned int m_alct_drift_delay;
  unsigned int m_alct_nplanes_hit_pretrig, m_alct_nplanes_hit_accel_pretrig;
  unsigned int m_alct_nplanes_hit_pattern, m_alct_nplanes_hit_accel_pattern;
  unsigned int m_alct_trig_mode, m_alct_accel_mode, m_alct_l1a_window_width;

  /** CLCT configuration parameters. */
  unsigned int m_clct_fifo_tbins, m_clct_fifo_pretrig;
  unsigned int m_clct_hit_persist, m_clct_drift_delay;
  unsigned int m_clct_nplanes_hit_pretrig, m_clct_nplanes_hit_pattern;
  unsigned int m_clct_pid_thresh_pretrig;
  unsigned int m_clct_min_separation;

  COND_SERIALIZABLE;
};

#endif
