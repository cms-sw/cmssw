//-------------------------------------------------
//
//   \class L1MuGMTParametersProducer
//
//   Description:  A class to produce the L1 GMT emulator parameters in the event setup
//
//
//   Author :
//   I. Mikulec
//
//--------------------------------------------------
#ifndef GMTConfigProducers_L1MuGMTParametersProducer_h
#define GMTConfigProducers_L1MuGMTParametersProducer_h

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1MuGMTParameters.h"
#include "CondFormats/DataRecord/interface/L1MuGMTParametersRcd.h"
#include "CondFormats/L1TObjects/interface/L1MuGMTChannelMask.h"
#include "CondFormats/DataRecord/interface/L1MuGMTChannelMaskRcd.h"

//
// class declaration
//

class L1MuGMTParametersProducer : public edm::ESProducer {
public:
  L1MuGMTParametersProducer(const edm::ParameterSet&);
  ~L1MuGMTParametersProducer() override;

  std::unique_ptr<L1MuGMTParameters> produceL1MuGMTParameters(const L1MuGMTParametersRcd&);
  std::unique_ptr<L1MuGMTChannelMask> produceL1MuGMTChannelMask(const L1MuGMTChannelMaskRcd&);

private:
  // ----------member data ---------------------------
  edm::ParameterSet* m_ps;
};

#endif
