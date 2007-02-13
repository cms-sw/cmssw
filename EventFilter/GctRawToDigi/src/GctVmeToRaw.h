#ifndef GctVmeToRaw_h
#define GctVmeToRaw_h

// -*- C++ -*-
//
// Package:    GctVmeToRaw
// Class:      GctVmeToRaw
// 
/**\class GctVmeToRaw GctVmeToRaw.cc EventFilter/GctRawToDigi/src/GctVmeToRaw.cc

 Description: Convert GCT VME output to Raw data format for unpacking

 Implementation:
     Input format is a 32 bit hex string per line (LSW first). Events separated with blank line.
*/
//
// Original Author:  Jim Brooke
//         Created:  Wed Nov  1 11:57:10 CET 2006
// $Id: GctVmeToRaw.h,v 1.1 2007/01/31 22:28:58 jbrooke Exp $
//
//


// system include files
#include <memory>
#include <string>
#include <fstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"

//
// class decleration
//

class GctVmeToRaw : public edm::EDProducer {
 public:
  explicit GctVmeToRaw(const edm::ParameterSet&);
  ~GctVmeToRaw();
  
 private: // methods
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

 private:
  
  int evtSize_; // store event size

  int fedId_;

  std::string filename_;
  std::ifstream file_;

};

#endif
