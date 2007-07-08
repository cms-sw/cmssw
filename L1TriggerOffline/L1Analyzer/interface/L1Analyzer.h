#ifndef L1Analyzer_L1Analyzer_h
#define L1Analyzer_L1Analyzer_h
// -*- C++ -*-
//
// Package:     L1Analyzer
// Class  :     L1Analyzer
// 
/**\class L1Analyzer L1Analyzer.h L1TriggerOffline/L1Analyzer/interface/L1Analyzer.h

 Description: Analyze the output of the GCT

 Usage:
    <usage>

*/
//
// Original Author:  Alex Tapper
//         Created:  Thu Nov 30 21:42:36 CET 2006
// $Id: L1Analyzer.h,v 1.1 2007/07/06 19:52:57 tapper Exp $
//

// user include files                                                                                         
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Histogram sets
#include "L1TriggerOffline/L1Analyzer/interface/SimpleHistograms.h"
#include "L1TriggerOffline/L1Analyzer/interface/ResolutionHistograms.h"
#include "L1TriggerOffline/L1Analyzer/interface/EfficiencyHistograms.h"

//                                                                                                            
// class declaration                                                                                          
//            
                                                                                                 
class L1Analyzer : public edm::EDAnalyzer {
 public:
  explicit L1Analyzer(const edm::ParameterSet&);
  ~L1Analyzer();

 private:
  virtual void analyze(const edm::Event&, const edm::EventSetup&);

  // ----------member data ---------------------------                                                     

  // Input sources

  edm::InputTag m_candidateSource;
  edm::InputTag m_referenceSource;
  edm::InputTag m_resMatchMapSource;
  edm::InputTag m_effMatchMapSource;

  // Different sets of histograms to make

  SimpleHistograms     *m_l1Simple, *m_refSimple;   // Simple kinematic distributions
  ResolutionHistograms *m_resolution; // Histograms for resolutions and biases
  EfficiencyHistograms *m_efficiency; // Histograms for efficiencies

};

#endif
