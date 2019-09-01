//-------------------------------------------------
//
//   \class L1MuTriggerPtScaleProducer
//
//   Description:  A class to produce the L1 mu emulator scales record in the event setup
//
//
//   Author :
//   W. Sun (copied from L1MuTriggerScalesProducer)
//
//--------------------------------------------------
#include "L1TriggerConfig/L1ScalesProducers/interface/L1MuTriggerPtScaleProducer.h"

L1MuTriggerPtScaleProducer::L1MuTriggerPtScaleProducer(const edm::ParameterSet& ps)
    : m_scales(ps.getParameter<int>("nbitPackingPt"),
               ps.getParameter<bool>("signedPackingPt"),
               ps.getParameter<int>("nbinsPt"),
               ps.getParameter<std::vector<double> >("scalePt")) {
  setWhatProduced(this, &L1MuTriggerPtScaleProducer::produceL1MuTriggerPtScale);
}

L1MuTriggerPtScaleProducer::~L1MuTriggerPtScaleProducer() {}

//
// member functions
//

// ------------ method called to produce the data  ------------
std::unique_ptr<L1MuTriggerPtScale> L1MuTriggerPtScaleProducer::produceL1MuTriggerPtScale(
    const L1MuTriggerPtScaleRcd& iRecord) {
  return std::make_unique<L1MuTriggerPtScale>(m_scales);
}
