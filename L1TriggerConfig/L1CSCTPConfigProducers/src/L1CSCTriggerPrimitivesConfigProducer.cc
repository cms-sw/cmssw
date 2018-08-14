//-----------------------------------------------------------------------------
//
//   Class: L1CSCTriggerPrimitivesConfigProducer
//
//   Description:
//
//   Author: Slava Valuev
//
//-----------------------------------------------------------------------------

#include <L1TriggerConfig/L1CSCTPConfigProducers/src/L1CSCTriggerPrimitivesConfigProducer.h>

#include "CondFormats/CSCObjects/interface/CSCDBL1TPParameters.h"
#include "CondFormats/DataRecord/interface/CSCDBL1TPParametersRcd.h"

//----------------
// Constructors --
//----------------

L1CSCTriggerPrimitivesConfigProducer::L1CSCTriggerPrimitivesConfigProducer(const edm::ParameterSet& iConfig) {
  // the following line is needed to tell the framework what
  // data is being produced
  setWhatProduced(this);

  // Decide on which of the two sets of parameters will be used.
  // (Temporary substitute for the IOV.)
  std::string alctParamSet, clctParamSet, tmbParamSet;
  alctParamSet = "alctParam";
  clctParamSet = "clctParam";
  tmbParamSet = "tmbParam";

  // get ALCT parameters from the config file
  edm::ParameterSet alctParams =
    iConfig.getParameter<edm::ParameterSet>(alctParamSet);
  m_alct_fifo_tbins   =
    alctParams.getParameter<unsigned int>("alctFifoTbins");
  m_alct_fifo_pretrig =
    alctParams.getParameter<unsigned int>("alctFifoPretrig");
  m_alct_drift_delay  =
    alctParams.getParameter<unsigned int>("alctDriftDelay");
  m_alct_nplanes_hit_pretrig =
    alctParams.getParameter<unsigned int>("alctNplanesHitPretrig");
  m_alct_nplanes_hit_pattern =
    alctParams.getParameter<unsigned int>("alctNplanesHitPattern");
  m_alct_nplanes_hit_accel_pretrig =
    alctParams.getParameter<unsigned int>("alctNplanesHitAccelPretrig");
  m_alct_nplanes_hit_accel_pattern =
    alctParams.getParameter<unsigned int>("alctNplanesHitAccelPattern");
  m_alct_trig_mode  =
    alctParams.getParameter<unsigned int>("alctTrigMode");
  m_alct_accel_mode =
    alctParams.getParameter<unsigned int>("alctAccelMode");
  m_alct_l1a_window_width =
    alctParams.getParameter<unsigned int>("alctL1aWindowWidth");

  // get CLCT parameters from the config file
  edm::ParameterSet clctParams =
    iConfig.getParameter<edm::ParameterSet>(clctParamSet);
  m_clct_fifo_tbins   =
    clctParams.getParameter<unsigned int>("clctFifoTbins");
  m_clct_fifo_pretrig =
    clctParams.getParameter<unsigned int>("clctFifoPretrig");
  m_clct_hit_persist  =
    clctParams.getParameter<unsigned int>("clctHitPersist");
  m_clct_drift_delay  =
    clctParams.getParameter<unsigned int>("clctDriftDelay");
  m_clct_nplanes_hit_pretrig =
    clctParams.getParameter<unsigned int>("clctNplanesHitPretrig");
  m_clct_nplanes_hit_pattern =
    clctParams.getParameter<unsigned int>("clctNplanesHitPattern");
  m_clct_pid_thresh_pretrig  =
    clctParams.getParameter<unsigned int>("clctPidThreshPretrig");
  m_clct_min_separation =
    clctParams.getParameter<unsigned int>("clctMinSeparation");

  // get TMB parameters from the config file
  edm::ParameterSet tmbParams =
    iConfig.getParameter<edm::ParameterSet>(tmbParamSet);
  m_tmb_mpc_block_me1a =
    tmbParams.getParameter<unsigned int>("tmbMpcBlockMe1a");
  m_tmb_alct_trig_enable =
    tmbParams.getParameter<unsigned int>("tmbAlctTrigEnable");
  m_tmb_clct_trig_enable =
    tmbParams.getParameter<unsigned int>("tmbClctTrigEnable");
  m_tmb_match_trig_enable =
    tmbParams.getParameter<unsigned int>("tmbMatchTrigEnable");
  m_tmb_match_trig_window_size =
    tmbParams.getParameter<unsigned int>("tmbMatchTrigWindowSize");
  m_tmb_tmb_l1a_window_size =
    tmbParams.getParameter<unsigned int>("tmbTmbL1aWindowSize");
}

//----------------
// Destructors  --
//----------------

L1CSCTriggerPrimitivesConfigProducer::~L1CSCTriggerPrimitivesConfigProducer() {
}

//------------------
// Member functions
//------------------

// ------------ method called to produce the data  ------------
std::unique_ptr<CSCDBL1TPParameters>
L1CSCTriggerPrimitivesConfigProducer::produce(const CSCDBL1TPParametersRcd& iRecord) {
  using namespace edm::es;
  //std::shared_ptr<L1CSCTriggerPrimitivesConfigProducer> pL1CSCTPConfigProducer;

  // Create empty collection of CSCTPParameters.
  auto pL1CSCTPParams = std::make_unique<CSCDBL1TPParameters>();

  // Set ALCT parameters.
  pL1CSCTPParams->setAlctFifoTbins(m_alct_fifo_tbins);
  pL1CSCTPParams->setAlctFifoPretrig(m_alct_fifo_pretrig);
  pL1CSCTPParams->setAlctDriftDelay(m_alct_drift_delay);
  pL1CSCTPParams->setAlctNplanesHitPretrig(m_alct_nplanes_hit_pretrig);
  pL1CSCTPParams->setAlctNplanesHitPattern(m_alct_nplanes_hit_pattern);
  pL1CSCTPParams->setAlctNplanesHitAccelPretrig(m_alct_nplanes_hit_accel_pretrig);
  pL1CSCTPParams->setAlctNplanesHitAccelPattern(m_alct_nplanes_hit_accel_pattern);
  pL1CSCTPParams->setAlctTrigMode(m_alct_trig_mode);
  pL1CSCTPParams->setAlctAccelMode(m_alct_accel_mode);
  pL1CSCTPParams->setAlctL1aWindowWidth(m_alct_l1a_window_width);

  // Set CLCT parameters.
  pL1CSCTPParams->setClctFifoTbins(m_clct_fifo_tbins);
  pL1CSCTPParams->setClctFifoPretrig(m_clct_fifo_pretrig);
  pL1CSCTPParams->setClctHitPersist(m_clct_hit_persist);
  pL1CSCTPParams->setClctDriftDelay(m_clct_drift_delay);
  pL1CSCTPParams->setClctNplanesHitPretrig(m_clct_nplanes_hit_pretrig);
  pL1CSCTPParams->setClctNplanesHitPattern(m_clct_nplanes_hit_pattern);
  pL1CSCTPParams->setClctPidThreshPretrig(m_clct_pid_thresh_pretrig);
  pL1CSCTPParams->setClctMinSeparation(m_clct_min_separation);

  // Set TMB parameters.
  pL1CSCTPParams->setTmbMpcBlockMe1a(m_tmb_mpc_block_me1a);
  pL1CSCTPParams->setTmbAlctTrigEnable(m_tmb_alct_trig_enable);
  pL1CSCTPParams->setTmbClctTrigEnable(m_tmb_clct_trig_enable);
  pL1CSCTPParams->setTmbMatchTrigEnable(m_tmb_match_trig_enable);
  pL1CSCTPParams->setTmbMatchTrigWindowSize(m_tmb_match_trig_window_size);
  pL1CSCTPParams->setTmbTmbL1aWindowSize(m_tmb_tmb_l1a_window_size);

  //return pL1CSCTPProducer;
  return pL1CSCTPParams;
}
