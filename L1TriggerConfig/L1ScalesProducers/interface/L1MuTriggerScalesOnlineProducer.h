//-------------------------------------------------
//
//   \class L1MuTriggerScalesOnlineProducer
//
//   Description:  A class to produce the L1 mu emulator scales record in the event setup
//                 from the OMDS database.
//
//   $Date: 2008/11/24 18:59:58 $
//   $Revision: 1.1 $
//
//   Author :
//   Thomas Themel
//
//--------------------------------------------------
#ifndef L1ScalesProducers_L1MuTriggerScalesOnlineProducer_h
#define L1ScalesProducers_L1MuTriggerScalesOnlineProducer_h

// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1MuTriggerScales.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerScalesRcd.h"
#include "CondTools/L1Trigger/interface/L1ConfigOnlineProdBase.h"


//
// class declaration
//

class L1MuTriggerScalesOnlineProducer : public L1ConfigOnlineProdBase<L1MuTriggerScalesRcd, L1MuTriggerScales> {
public:
  L1MuTriggerScalesOnlineProducer(const edm::ParameterSet&);
  ~L1MuTriggerScalesOnlineProducer();

  virtual boost::shared_ptr<L1MuTriggerScales> newObject(
	const std::string& objectKey ) ;

private:
  // ----------member data ---------------------------
  L1MuTriggerScales m_scales;
  unsigned int m_nbitPackingPhi;
  unsigned int m_nbitPackingEta; 
  unsigned int m_nbinsEta;
  bool m_signedPackingPhi;

};

#endif
