#ifndef L1RCTInputProducer_h
#define L1RCTInputProducer_h

#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/DataRecord/interface/L1CaloEcalScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1CaloHcalScaleRcd.h"
#include "CondFormats/L1TObjects/interface/L1CaloEcalScale.h"
#include "CondFormats/L1TObjects/interface/L1CaloHcalScale.h"
#include "CondFormats/DataRecord/interface/L1EmEtScaleRcd.h"
#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"
#include "CondFormats/DataRecord/interface/L1RCTParametersRcd.h"
#include "CondFormats/L1TObjects/interface/L1RCTParameters.h"
#include "CondFormats/DataRecord/interface/L1RCTChannelMaskRcd.h"
#include "CondFormats/L1TObjects/interface/L1RCTChannelMask.h"

#include <string>

class L1RCT;
class L1RCTLookupTables;

class L1RCTInputProducer : public edm::stream::EDProducer<> {
public:
  explicit L1RCTInputProducer(const edm::ParameterSet &ps);
  ~L1RCTInputProducer() override;
  void produce(edm::Event &e, const edm::EventSetup &c) override;

private:
  L1RCTLookupTables *rctLookupTables;
  L1RCT *rct;
  bool useEcal;
  bool useHcal;
  edm::InputTag ecalDigisLabel;
  edm::InputTag hcalDigisLabel;
  edm::ESGetToken<L1RCTParameters, L1RCTParametersRcd> rctParametersToken;
  edm::ESGetToken<L1RCTChannelMask, L1RCTChannelMaskRcd> channelMaskToken;
  edm::ESGetToken<L1CaloEcalScale, L1CaloEcalScaleRcd> ecalScaleToken;
  edm::ESGetToken<L1CaloHcalScale, L1CaloHcalScaleRcd> hcalScaleToken;
  edm::ESGetToken<L1CaloEtScale, L1EmEtScaleRcd> emScaleToken;
};
#endif
