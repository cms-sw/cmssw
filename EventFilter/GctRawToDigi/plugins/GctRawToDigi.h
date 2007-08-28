#ifndef GctRawToDigi_h
#define GctRawToDigi_h

// -*- C++ -*-
//
// Package:    GctRawToDigi
// Class:      GctRawToDigi
// 
/**\class GctRawToDigi GctRawToDigi.cc EventFilter/GctRawToDigi/src/GctRawToDigi.cc

 Description: Produce GCT digis from raw data

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jim Brooke
//         Created:  Wed Nov  1 11:57:10 CET 2006
// $Id: GctRawToDigi.h,v 1.8 2007/07/24 20:19:40 jbrooke Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"

#include "EventFilter/GctRawToDigi/src/GctBlockUnpacker.h"

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

 private: // members

  static unsigned MAX_EXCESS;
  static unsigned MAX_BLOCKS;

  bool verbose_;         // print out for each event

  edm::InputTag inputLabel_;  // FED collection label

  int fedId_;            // GCT FED ID
  int nDebugSamples_;    // number of samples per block in debug mode
  
  // unpacking options
  bool doEm_;
  bool doJets_;
  bool doEtSums_;
  bool doInternEm_;
  bool doFibres_;


  // Block to Digi converter
  GctBlockUnpacker blockUnpacker_;

};

#endif
