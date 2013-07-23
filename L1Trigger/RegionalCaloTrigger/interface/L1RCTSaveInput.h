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
// $Id: L1RCTSaveInput.h,v 1.6 2012/02/09 13:12:23 eulisse Exp $
//
//

#include <memory>
#include <iostream>
#include <fstream>

#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/MakerMacros.h"

class L1RCTLookupTables;
class L1RCT;

class L1RCTSaveInput : public edm::EDAnalyzer {
public:
  explicit L1RCTSaveInput(const edm::ParameterSet&);
  ~L1RCTSaveInput();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
private:
  std::string fileName;
  L1RCTLookupTables* rctLookupTables;
  L1RCT* rct;
  bool useEcal;
  bool useHcal;
  edm::InputTag ecalDigisLabel;
  edm::InputTag hcalDigisLabel;
  bool useDebugTpgScales;
  std::ofstream ofs;
};
