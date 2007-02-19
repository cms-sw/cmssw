// Analyzer Plug: module does nothing and might be useful in case user simply 
// wants to extract some number of events from ROOT tuple.
//
// Author : Samvel Khalatian ( samvel at fnal dot gov)
// Created: 02/19/07
// Licence: GPL

#ifndef ANALYSISEXAMPLES_SISTRIPDETECTORPERFORMANCE_ANALYZERPLUG_H
#define ANALYSISEXAMPLES_SISTRIPDETECTORPERFORMANCE_ANALYZERPLUG_H

#include "FWCore/Framework/interface/EDAnalyzer.h"

// Save Compile time by forwarding declarations
#include "FWCore/Framework/interface/Frameworkfwd.h"

class AnalyzerPlug: public edm::Analyzer {
  public:
    // Constructor
    inline AnalyzerPlug( const edm::ParameterSet &roCONFIG):
      edm::Analyzer( roCONFIG) {}

    // Destructor
    inline virtual ~AnalyzerPlug() {}

    // Default copying is used
  protected:
    inline virtual void analyze( const edm::Event &roEVENT,
                                 const edm::EventSetup &roEVENT_SETUP) {}
};

#endif // ANALYSISEXAMPLES_SISTRIPDETECTORPERFORMANCE_ANALYZERPLUG_H
