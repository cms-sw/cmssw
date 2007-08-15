#ifndef L1TriggerConfig_L1CSCTriggerPrimitivesConfigProducer_h
#define L1TriggerConfig_L1CSCTriggerPrimitivesConfigProducer_h

/** \class L1CSCTriggerPrimitivesConfigProducer
 *
 * Description: Produce configuration parameters for the Level-1 CSC Trigger
 *              Primitives emulator.
 *
 * \author Slava Valuev
 * Created: Thu Apr 12 11:26:54 CEST 2007
 * $Id: L1CSCTriggerPrimitivesConfigProducer.h,v 1.1 2007/04/17 13:30:12 slava Exp $
 *
 */

// system include files
#include <memory>
//#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

class L1CSCTPParameters;
class L1CSCTPParametersRcd;

class L1CSCTriggerPrimitivesConfigProducer : public edm::ESProducer {
 public:
  L1CSCTriggerPrimitivesConfigProducer(const edm::ParameterSet&);
  ~L1CSCTriggerPrimitivesConfigProducer();

  //typedef boost::shared_ptr<L1CSCTriggerPrimitivesConfigProducer> ReturnType;

  std::auto_ptr<L1CSCTPParameters> produce(const L1CSCTPParametersRcd&);

 private:
  /** ALCT configuration parameters. */
  unsigned int m_alct_fifo_tbins, m_alct_fifo_pretrig;
  unsigned int m_alct_bx_width,   m_alct_drift_delay;
  unsigned int m_alct_nph_thresh, m_alct_nph_pattern;
  unsigned int m_alct_trig_mode,  m_alct_alct_amode, m_alct_l1a_window;

  /** CLCT configuration parameters. */
  unsigned int m_clct_fifo_tbins, m_clct_fifo_pretrig;
  unsigned int m_clct_bx_width,   m_clct_drift_delay;
  unsigned int m_clct_nph_pattern;
  unsigned int m_clct_hs_thresh,  m_clct_ds_thresh;
};

#endif
