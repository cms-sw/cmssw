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
// $Id: GctDigiToRaw.h,v 1.5 2007/10/08 19:54:32 frazier Exp $
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

#include "EventFilter/GctRawToDigi/src/GctBlockPacker.h"

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
  
  void print(FEDRawData& data);

 private:  // members

  // input tags
  edm::InputTag rctInputLabel_;
  edm::InputTag gctInputLabel_;

  // pack flags
  bool packRctEm_;
  bool packRctCalo_;

  // FED numbers
  int fedId_;            

  // print out for each event
  bool verbose_;

  // counter events
  int counter_;          
  
  // digi to block converter
  GctBlockPacker blockPacker_;

};

#endif
