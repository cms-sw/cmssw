// -*- C++ -*-
//
// Package:    L1RCTSaveInput
// Class:      L1RCTSaveInput
//
/**\class L1RCTSaveInput L1RCTSaveInput.cc src/L1RCTSaveInput/src/L1RCTSaveInput.cc

 Description: Saves the input event from TPGs for loading 
              simulated events in hardware

 Implementation: Kind of kludgy -- should think of a better way in future

*/
//
// Original Author:  Sridhara Dasu
//         Created:  Tue Jul 31 17:10:13 CEST 2007
// $Id$
//
//

#include <memory>
#include <iostream>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class L1RCTSaveInput : public edm::EDAnalyzer {
public:
  explicit L1RCTSaveInput(const edm::ParameterSet&);
  ~L1RCTSaveInput();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
private:
  std::string fileName;
  L1RCTLookupTables* rctLookupTables;
  L1RCT* rct;
  std::ofstream ofs;
};
