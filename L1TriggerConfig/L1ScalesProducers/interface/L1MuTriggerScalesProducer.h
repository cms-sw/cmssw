//-------------------------------------------------
//
//   \class L1MuTriggerScalesProducer
//
//   Description:  A class to produce the L1 mu emulator scales record in the event setup
//
//   $Date: 2008/04/17 23:33:41 $
//   $Revision: 1.2 $
//
//   Author :
//   I. Mikulec
//
//--------------------------------------------------
#ifndef L1ScalesProducers_L1MuTriggerScalesProducer_h
#define L1ScalesProducers_L1MuTriggerScalesProducer_h

// system include files
#include <memory>
#include <boost/shared_ptr.hpp>
#include <vector>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1MuTriggerScales.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerScalesRcd.h"


//
// class declaration
//

class L1MuTriggerScalesProducer : public edm::ESProducer {
public:
  L1MuTriggerScalesProducer(const edm::ParameterSet&);
  ~L1MuTriggerScalesProducer();
  
  std::auto_ptr<L1MuTriggerScales> produceL1MuTriggerScales(const L1MuTriggerScalesRcd&);

private:
  // ----------member data ---------------------------
  
  L1MuTriggerScales m_scales ;
};

#endif
