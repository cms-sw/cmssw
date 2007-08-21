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

#include "CondFormats/L1TObjects/interface/L1CSCTPParameters.h"
#include "CondFormats/DataRecord/interface/L1CSCTPParametersRcd.h"

//----------------
// Constructors --
//----------------

L1CSCTriggerPrimitivesConfigProducer::L1CSCTriggerPrimitivesConfigProducer(const edm::ParameterSet& iConfig) {
  // the following line is needed to tell the framework what
  // data is being produced
  setWhatProduced(this);

  // Decide on which of the two sets of parameters will be used.
  // (Temporary substitute for the IOV.)
  std::string alctParamSet, clctParamSet;
  bool isMTCC = iConfig.getParameter<bool>("isMTCC");
  if (isMTCC) {
    alctParamSet = "alctParamMTCC2";
    clctParamSet = "clctParamMTCC2";
  }
  else {
    alctParamSet = "alctParam";
    clctParamSet = "clctParam";
  }

  // get ALCT parameters from the config file
  edm::ParameterSet alctParams =
    iConfig.getParameter<edm::ParameterSet>(alctParamSet);
  m_alct_fifo_tbins  = alctParams.getParameter<unsigned int>("alctFifoTbins");
  m_alct_fifo_pretrig= alctParams.getParameter<unsigned int>("alctFifoPretrig");
  m_alct_bx_width    = alctParams.getParameter<unsigned int>("alctBxWidth");
  m_alct_drift_delay = alctParams.getParameter<unsigned int>("alctDriftDelay");
  m_alct_nph_thresh  = alctParams.getParameter<unsigned int>("alctNphThresh");
  m_alct_nph_pattern = alctParams.getParameter<unsigned int>("alctNphPattern");
  m_alct_trig_mode   = alctParams.getParameter<unsigned int>("alctTrigMode");
  m_alct_alct_amode  = alctParams.getParameter<unsigned int>("alctAlctAmode");
  m_alct_l1a_window  = alctParams.getParameter<unsigned int>("alctL1aWindow");

  // get CLCT parameters from the config file
  edm::ParameterSet clctParams =
    iConfig.getParameter<edm::ParameterSet>(clctParamSet);
  m_clct_fifo_tbins  = clctParams.getParameter<unsigned int>("clctFifoTbins");
  m_clct_fifo_pretrig= clctParams.getParameter<unsigned int>("clctFifoPretrig");
  m_clct_bx_width    = clctParams.getParameter<unsigned int>("clctBxWidth");
  m_clct_drift_delay = clctParams.getParameter<unsigned int>("clctDriftDelay");
  m_clct_nph_pattern = clctParams.getParameter<unsigned int>("clctNphPattern");
  m_clct_hs_thresh   = clctParams.getParameter<unsigned int>("clctHsThresh");
  m_clct_ds_thresh   = clctParams.getParameter<unsigned int>("clctDsThresh");
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
std::auto_ptr<L1CSCTPParameters>
L1CSCTriggerPrimitivesConfigProducer::produce(const L1CSCTPParametersRcd& iRecord) {
  using namespace edm::es;
  //boost::shared_ptr<L1CSCTriggerPrimitivesConfigProducer> pL1CSCTPConfigProducer;

  // Create empty collection of CSCTPParameters.
  std::auto_ptr<L1CSCTPParameters> pL1CSCTPParams(new L1CSCTPParameters);

  // Set ALCT parameters.
  pL1CSCTPParams->setAlctFifoTbins(m_alct_fifo_tbins);
  pL1CSCTPParams->setAlctFifoPretrig(m_alct_fifo_pretrig);
  pL1CSCTPParams->setAlctBxWidth(m_alct_bx_width);
  pL1CSCTPParams->setAlctDriftDelay(m_alct_drift_delay);
  pL1CSCTPParams->setAlctNphThresh(m_alct_nph_thresh);
  pL1CSCTPParams->setAlctNphPattern(m_alct_nph_pattern);
  pL1CSCTPParams->setAlctTrigMode(m_alct_trig_mode);
  pL1CSCTPParams->setAlctAlctAmode(m_alct_alct_amode);
  pL1CSCTPParams->setAlctL1aWindow(m_alct_l1a_window);

  // Set CLCT parameters.
  pL1CSCTPParams->setClctFifoTbins(m_clct_fifo_tbins);
  pL1CSCTPParams->setClctFifoPretrig(m_clct_fifo_pretrig);
  pL1CSCTPParams->setClctBxWidth(m_clct_bx_width);
  pL1CSCTPParams->setClctDriftDelay(m_clct_drift_delay);
  pL1CSCTPParams->setClctNphPattern(m_clct_nph_pattern);
  pL1CSCTPParams->setClctHsThresh(m_clct_hs_thresh);
  pL1CSCTPParams->setClctDsThresh(m_clct_ds_thresh);

  //return pL1CSCTPProducer;
  return pL1CSCTPParams;
}
