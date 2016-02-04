//-------------------------------------------------
//
//   \class L1MuTriggerPtScaleOnlineProducer
//
//   Description:  A class to produce the L1 mu emulator scales record in the event setup
//
//   $Date: 2008/11/24 18:59:58 $
//   $Revision: 1.1 $
//
//   Author :
//   W. Sun (copied from L1MuTriggerScalesProducer)
//
//--------------------------------------------------
#ifndef L1ScalesProducers_L1MuTriggerPtScaleOnlineProducer_h
#define L1ScalesProducers_L1MuTriggerPtScaleOnlineProducer_h

// system include files
#include <memory>
#include <vector>

// user include files
#include "CondTools/L1Trigger/interface/L1ConfigOnlineProdBase.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "L1TriggerConfig/L1ScalesProducers/interface/ScaleRecordHelper.h"
#include "CondFormats/L1TObjects/interface/L1MuTriggerPtScale.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerPtScaleRcd.h"


//
// class declaration
//

class L1MuTriggerPtScaleOnlineProducer : public L1ConfigOnlineProdBase<L1MuTriggerPtScaleRcd, L1MuTriggerPtScale> {
public:
  L1MuTriggerPtScaleOnlineProducer(const edm::ParameterSet&);
  ~L1MuTriggerPtScaleOnlineProducer();
  
  boost::shared_ptr<L1MuTriggerPtScale> newObject(const std::string& objectKey);

private:
  // ----------member data ---------------------------
  
  bool m_signedPacking; 
  unsigned int m_nbitsPacking;
  unsigned int m_nBins;

};

#endif
