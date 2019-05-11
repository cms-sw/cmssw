#ifndef L1TriggerConfig_L1CSCTriggerPrimitivesConfigProducer_h
#define L1TriggerConfig_L1CSCTriggerPrimitivesConfigProducer_h

/** \class L1CSCTriggerPrimitivesConfigProducer
 *
 * Description: Produce configuration parameters for the Level-1 CSC Trigger
 *              Primitives emulator.
 *
 * \author Slava Valuev
 * Created: Thu Apr 12 11:26:54 CEST 2007
 *
 */

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

class CSCDBL1TPParameters;
class CSCDBL1TPParametersRcd;

class L1CSCTriggerPrimitivesConfigProducer : public edm::ESProducer {
public:
  L1CSCTriggerPrimitivesConfigProducer(const edm::ParameterSet&);
  ~L1CSCTriggerPrimitivesConfigProducer() override;

  //typedef std::shared_ptr<L1CSCTriggerPrimitivesConfigProducer> ReturnType;

  std::unique_ptr<CSCDBL1TPParameters> produce(const CSCDBL1TPParametersRcd&);

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

  /** TMB configuration parameters. */
  unsigned int m_tmb_mpc_block_me1a;
  unsigned int m_tmb_alct_trig_enable, m_tmb_clct_trig_enable;
  unsigned int m_tmb_match_trig_enable;
  unsigned int m_tmb_match_trig_window_size, m_tmb_tmb_l1a_window_size;
};

#endif
