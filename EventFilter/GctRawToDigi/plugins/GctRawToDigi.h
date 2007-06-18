#ifndef GctRawToDigi_h
#define GctRawToDigi_h

// -*- C++ -*-
//
// Package:    GctRawToDigi
// Class:      GctRawToDigi
// 
/**\class GctRawToDigi GctRawToDigi.cc GctRawToDigi/GctRawToDigi/src/GctRawToDigi.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jim Brooke
//         Created:  Wed Nov  1 11:57:10 CET 2006
// $Id: GctRawToDigi.h,v 1.7 2007/06/07 13:32:43 jbrooke Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"

#include "EventFilter/GctRawToDigi/src/GctBlockConverter.h"

//
// class decleration
//

class GctRawToDigi : public edm::EDProducer {
 public:
  explicit GctRawToDigi(const edm::ParameterSet&);
  ~GctRawToDigi();
  
 private: // methods
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  void unpack(const FEDRawData& d, edm::Event& e);

 private:  // members

  static unsigned MAX_EXCESS;
  static unsigned MAX_BLOCKS;

  bool verbose_;         // print out for each event

  int fedId_;            // GCT FED ID
  int nDebugSamples_;    // number of samples per block in debug mode
  
  // Block to Digi converter
  GctBlockConverter converter_;

};

#endif
