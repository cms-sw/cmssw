#ifndef L1GCTANALYZER_FIBREANALYZER_H
#define L1GCTANALYZER_FIBREANALYZER_H

// -*- C++ -*-
//
// Package:    GctFibreAnalyzer
// Class:      GctFibreAnalyzer
//
/**\class GctFibreAnalyzer GctFibreAnalyzer.cc L1Trigger/L1GctAnalzyer/interface/GctFibreAnalyzer.h

Description: Analyzer individual fibre channels from the source card.

*/
//
// Original Author:  Alex Tapper
//         Created:  Thu Jul 12 14:21:06 CEST 2007
//
//

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

// Gct fibre data format
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctFibreWord.h"

class GctFibreAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit GctFibreAnalyzer(const edm::ParameterSet&);
  ~GctFibreAnalyzer() override;

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  bool CheckFibreWord(const L1GctFibreWord fibre);
  bool CheckForBC0(const L1GctFibreWord fibre);
  void CheckLogicalID(const L1GctFibreWord fibre);
  void CheckCounter(const L1GctFibreWord fibre);

  edm::InputTag m_fibreSource;
  bool m_doLogicalID;
  bool m_doCounter;
  unsigned int m_numZeroEvents;
  unsigned int m_numInconsistentPayloadEvents;
  unsigned int m_numConsistentEvents;
};

#endif
