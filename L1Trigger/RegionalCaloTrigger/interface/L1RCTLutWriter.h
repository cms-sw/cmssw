#ifndef L1RCTLutWriter_h
#define L1RCTLutWriter_h

// -*- C++ -*-
//
// Package:    L1RCTLutWriter
// Class:      L1RCTLutWriter
//
/**\class L1RCTLutWriter L1RCTLutWriter.cc L1RCTLutWriter.h
 L1Trigger/L1RCTLutWriter/src/L1RCTLutWriter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  jleonard
//         Created:  Fri Apr 11 16:27:07 CEST 2008
//
//

// system include files
#include <memory>

#include <fstream>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "FWCore/Framework/interface/ESHandle.h"    // why doesn't mkedanlzr
#include "FWCore/Framework/interface/EventSetup.h"  // add these??
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/DataRecord/interface/L1RCTParametersRcd.h"
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
class L1RCTLookupTables;
class L1RCTParameters;
// class L1RCTChannelMask;

//
// class declaration
//

class L1RCTLutWriter : public edm::one::EDAnalyzer<> {
public:
  explicit L1RCTLutWriter(const edm::ParameterSet &);
  ~L1RCTLutWriter() override;

private:
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void endJob() override;
  void writeRcLutFile(unsigned short card);
  void writeEicLutFile(unsigned short card);
  void writeJscLutFile();
  void writeThresholdsFile(unsigned int eicThreshold, unsigned int jscThresholdBarrel, unsigned int jscThresholdEndcap);

  // ----------member data ---------------------------

  L1RCTLookupTables *lookupTable_;
  const L1RCTParameters *rctParameters_;
  // const L1RCTChannelMask* channelMask_;
  std::ofstream lutFile_;
  std::string keyName_;
  edm::ESGetToken<L1RCTParameters, L1RCTParametersRcd> rctParametersToken_;
  edm::ESGetToken<L1CaloEtScale, L1EmEtScaleRcd> emScaleToken_;
  edm::ESGetToken<CaloTPGTranscoder, CaloTPGRecord> transcoderToken_;
  edm::ESGetToken<L1CaloHcalScale, L1CaloHcalScaleRcd> hcalScaleToken_;
  edm::ESGetToken<L1CaloEcalScale, L1CaloEcalScaleRcd> ecalScaleToken_;
  bool useDebugTpgScales_;
  EcalTPGScale::Tokens tokens_;
};
#endif
