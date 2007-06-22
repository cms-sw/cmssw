#ifndef GctDigiToRaw_h
#define GctDigiToRaw_h

// -*- C++ -*-
//
// Package:    GctDigiToRaw
// Class:      GctDigiToRaw
// 
/**\class GctDigiToRaw GctDigiToRaw.cc EventFilter/GctRawToDigi/src/GctDigiToRaw.cc

 Description: Produce fake GCT raw data from digis

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jim Brooke
//         Created:  Wed Nov  1 11:57:10 CET 2006
// $Id: GctDigiToRaw.h,v 1.4 2007/06/18 12:12:34 jbrooke Exp $
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

#include "EventFilter/GctRawToDigi/src/GctBlockConverter.h"

//
// class decleration
//

class GctDigiToRaw : public edm::EDProducer {
 public:
  explicit GctDigiToRaw(const edm::ParameterSet&);
  ~GctDigiToRaw();
  
 private: // methods
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  void unpack(const FEDRawData& d, edm::Event& e);

 private:  // members

  bool verbose_;         // print out for each event

  edm::InputTag inputLabel_;

  int fedId_;            // GCT FED ID
  int nDebugSamples_;    // number of samples per block in debug mode
  
  // Block to Digi converter
  GctBlockConverter converter_;

};

#endif
