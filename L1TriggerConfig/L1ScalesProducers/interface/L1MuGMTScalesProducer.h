//-------------------------------------------------
//
//   \class L1MuGMTScalesProducer
//
//   Description:  A class to produce the L1 GMT emulator scales record in the event setup
//
//   $Date: 2008/04/17 23:33:41 $
//   $Revision: 1.2 $
//
//   Author :
//   I. Mikulec
//
//--------------------------------------------------
#ifndef L1ScalesProducers_L1MuGMTScalesProducer_h
#define L1ScalesProducers_L1MuGMTScalesProducer_h

// system include files
#include <memory>
#include <boost/shared_ptr.hpp>
#include <vector>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1MuGMTScales.h"
#include "CondFormats/DataRecord/interface/L1MuGMTScalesRcd.h"


//
// class declaration
//

class L1MuGMTScalesProducer : public edm::ESProducer {
public:
  L1MuGMTScalesProducer(const edm::ParameterSet&);
  ~L1MuGMTScalesProducer();
  
  std::auto_ptr<L1MuGMTScales> produceL1MuGMTScales(const L1MuGMTScalesRcd&);

private:
  // ----------member data ---------------------------
  
  L1MuGMTScales m_scales ;
};

#endif
