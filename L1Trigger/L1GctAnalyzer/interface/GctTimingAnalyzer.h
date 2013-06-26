#ifndef L1GCTANALYZER_TIMINGANALYZER_H
#define L1GCTANALYZER_TIMINGANALYZER_H

// -*- C++ -*-
//
// Package:    GctTimingAnalyzer
// Class:      GctTimingAnalyzer
// 
/**\class GctTimingAnalyzer GctTimingAnalyzer.cc L1Trigger/L1GctAnalzyer/interface/GctTimingAnalyzer.h

Description: Analyse the timing of all of the GCT pipelines

*/
//
// Original Author:  Alex Tapper
//         Created:  Mon Apr 21 14:21:06 CEST 2008
// $Id: GctTimingAnalyzer.h,v 1.6 2008/09/04 16:57:38 tapper Exp $
//
//

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

// Data formats
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"

#include <iostream>
#include <fstream>

class GctTimingAnalyzer : public edm::EDAnalyzer {

 public:

  explicit GctTimingAnalyzer(const edm::ParameterSet&);
  ~GctTimingAnalyzer();

 private:

  virtual void analyze(const edm::Event&, const edm::EventSetup&);

  std::string m_outputFileName; // Output file
  std::ofstream m_outputFile;

  edm::InputTag m_gctSource; // General source label
  edm::InputTag m_isoEmSource;   
  edm::InputTag m_nonIsoEmSource;
  edm::InputTag m_cenJetsSource;
  edm::InputTag m_forJetsSource;
  edm::InputTag m_tauJetsSource;

  bool m_doInternal;  // Do internal pipelines
  bool m_doElectrons;
  bool m_doJets;
  bool m_doHFRings;
  bool m_doESums;

  unsigned m_evtNum;

};

#endif
