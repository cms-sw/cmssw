#ifndef L1GCTPRINTLUTS_H
#define L1GCTPRINTLUTS_H
// -*- C++ -*-
//
// Package:    L1GlobalCaloTrigger
// Class:      L1GctPrintLuts
//
/**\class L1GctPrintLuts L1GctPrintLuts.cc L1Trigger/L1GlobalCaloTrigger/plugins/L1GctPrintLuts.cc

 Description: print Gct lookup table contents to a file

*/
//
// Author: Greg Heath
// Date:   July 2008
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>

#include "L1Trigger/GlobalCaloTrigger/interface/L1GlobalCaloTrigger.h"

// Trigger configuration includes
#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"
#include "CondFormats/L1TObjects/interface/L1GctJetFinderParams.h"
#include "CondFormats/DataRecord/interface/L1JetEtScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1HtMissScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1HfRingEtScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1GctJetFinderParamsRcd.h"

//
// class declaration
//

class L1GctPrintLuts : public edm::EDAnalyzer {
public:
  /// typedefs
  typedef L1GlobalCaloTrigger::lutPtr lutPtr;
  typedef L1GlobalCaloTrigger::lutPtrVector lutPtrVector;

  explicit L1GctPrintLuts(const edm::ParameterSet&);
  ~L1GctPrintLuts() override;

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  int configureGct(const edm::EventSetup& c);

  // ----------member data ---------------------------

  // output file names
  std::string m_jetRanksOutFileName;
  std::string m_hfSumLutOutFileName;
  std::string m_htMissLutOutFileName;

  // pointer to the actual emulator
  L1GlobalCaloTrigger* m_gct;

  // pointers to the jet Et LUTs
  lutPtrVector m_jetEtCalibLuts;

  //EventSetup Tokens
  edm::ESGetToken<L1GctJetFinderParams, L1GctJetFinderParamsRcd> m_jfParsToken;
  edm::ESGetToken<L1CaloEtScale, L1JetEtScaleRcd> m_etScaleToken;
  edm::ESGetToken<L1CaloEtScale, L1HtMissScaleRcd> m_htMissScaleToken;
  edm::ESGetToken<L1CaloEtScale, L1HfRingEtScaleRcd> m_hfRingEtScaleToken;
};
#endif
