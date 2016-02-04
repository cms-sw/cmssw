#ifndef L1RCTLutWriter_h
#define L1RCTLutWriter_h

// -*- C++ -*-
//
// Package:    L1RCTLutWriter
// Class:      L1RCTLutWriter
// 
/**\class L1RCTLutWriter L1RCTLutWriter.cc L1RCTLutWriter.h L1Trigger/L1RCTLutWriter/src/L1RCTLutWriter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  jleonard
//         Created:  Fri Apr 11 16:27:07 CEST 2008
// $Id: L1RCTLutWriter.h,v 1.5 2010/08/07 14:55:58 wmtan Exp $
//
//


// system include files
#include <memory>

#include <iostream>
#include <fstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"   // why doesn't mkedanlzr
#include "FWCore/Framework/interface/EventSetup.h" // add these??

class L1RCTLookupTables;
class L1RCTParameters;
//class L1RCTChannelMask;

//
// class declaration
//

class L1RCTLutWriter : public edm::EDAnalyzer {
public:
  explicit L1RCTLutWriter(const edm::ParameterSet&);
  ~L1RCTLutWriter();
  
  
private:
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  void writeRcLutFile(unsigned short card);
  void writeEicLutFile(unsigned short card);
  void writeJscLutFile();
  void writeThresholdsFile(unsigned int eicThreshold,
			   unsigned int jscThresholdBarrel,
			   unsigned int jscThresholdEndcap);
  
  // ----------member data ---------------------------
  
  L1RCTLookupTables* lookupTable_;
  const L1RCTParameters* rctParameters_;
  //const L1RCTChannelMask* channelMask_;
  std::ofstream lutFile_;
  std::string keyName_;
  bool useDebugTpgScales_;

};
#endif
