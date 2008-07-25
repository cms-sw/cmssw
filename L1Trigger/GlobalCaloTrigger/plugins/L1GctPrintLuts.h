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

//
// class declaration
//

class L1GlobalCaloTrigger;
class L1GctJetEtCalibrationLut;


class L1GctPrintLuts : public edm::EDAnalyzer {
 public:
  explicit L1GctPrintLuts(const edm::ParameterSet&);
  ~L1GctPrintLuts();


 private:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  // ----------member data ---------------------------

  void configureGct(const edm::EventSetup& c) ;

  // output file name
  std::string m_outputFileName;

  // pointer to the actual emulator
  L1GlobalCaloTrigger* m_gct;

  // pointer to the jet Et LUT
  L1GctJetEtCalibrationLut* m_jetEtCalibLut;

};
#endif
