//-------------------------------------------------
//
//   \class L1MuTriggerPtScaleProducer
//
//   Description:  A class to produce the L1 mu emulator scales record in the event setup
//
//   $Date: 2008/04/17 23:33:09 $
//   $Revision: 1.1 $
//
//   Author :
//   W. Sun (copied from L1MuTriggerScalesProducer)
//
//--------------------------------------------------
#ifndef L1ScalesProducers_L1MuTriggerPtScaleProducer_h
#define L1ScalesProducers_L1MuTriggerPtScaleProducer_h

// system include files
#include <memory>
#include <boost/shared_ptr.hpp>
#include <vector>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1MuTriggerPtScale.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerPtScaleRcd.h"


//
// class declaration
//

class L1MuTriggerPtScaleProducer : public edm::ESProducer {
public:
  L1MuTriggerPtScaleProducer(const edm::ParameterSet&);
  ~L1MuTriggerPtScaleProducer();
  
  std::auto_ptr<L1MuTriggerPtScale> produceL1MuTriggerPtScale(const L1MuTriggerPtScaleRcd&);

private:
  // ----------member data ---------------------------
  
  L1MuTriggerPtScale m_scales ;
};

#endif
