// -*- C++ -*-
//
// Package:    L1RCTSaveInput
// Class:      L1RCTSaveInput
//
/**\class L1RCTSaveInput L1RCTSaveInput.cc
 src/L1RCTSaveInput/src/L1RCTSaveInput.cc

 Description: Saves the input event from TPGs for loading
              simulated events in hardware

 Implementation: Kind of kludgy -- should think of a better way in future

*/
//
// Original Author:  Sridhara Dasu
//         Created:  Tue Jul 31 17:10:13 CEST 2007
//
//

#include <fstream>
#include <iostream>
#include <memory>

#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "CondFormats/DataRecord/interface/L1RCTChannelMaskRcd.h"
#include "CondFormats/DataRecord/interface/L1RCTParametersRcd.h"
#include "CondFormats/L1TObjects/interface/L1RCTChannelMask.h"
#include "CondFormats/L1TObjects/interface/L1RCTParameters.h"
#include "CondFormats/DataRecord/interface/L1CaloEcalScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1CaloHcalScaleRcd.h"
#include "CondFormats/L1TObjects/interface/L1CaloEcalScale.h"
#include "CondFormats/L1TObjects/interface/L1CaloHcalScale.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGRecord.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"
#include "CondFormats/DataRecord/interface/L1EmEtScaleRcd.h"
#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"
#include "CalibCalorimetry/EcalTPGTools/interface/EcalTPGScale.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

class L1RCTLookupTables;
class L1RCT;

class L1RCTSaveInput : public edm::one::EDAnalyzer<> {
public:
  explicit L1RCTSaveInput(const edm::ParameterSet &);
  ~L1RCTSaveInput() override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;

private:
  std::string fileName;
  L1RCTLookupTables *rctLookupTables;
  L1RCT *rct;
  bool useEcal;
  bool useHcal;
  edm::EDGetTokenT<EcalTrigPrimDigiCollection> ecalDigisToken_;
  edm::EDGetTokenT<HcalTrigPrimDigiCollection> hcalDigisToken_;
  edm::ESGetToken<L1RCTParameters, L1RCTParametersRcd> rctParametersToken_;
  edm::ESGetToken<L1RCTChannelMask, L1RCTChannelMaskRcd> channelMaskToken_;
  edm::ESGetToken<L1CaloEtScale, L1EmEtScaleRcd> emScaleToken_;
  edm::ESGetToken<CaloTPGTranscoder, CaloTPGRecord> transcoderToken_;
  edm::ESGetToken<L1CaloHcalScale, L1CaloHcalScaleRcd> hcalScaleToken_;
  edm::ESGetToken<L1CaloEcalScale, L1CaloEcalScaleRcd> ecalScaleToken_;
  bool useDebugTpgScales;
  std::ofstream ofs;
  EcalTPGScale::Tokens tokens_;
};
