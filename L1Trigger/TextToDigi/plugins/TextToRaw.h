#ifndef TextToRaw_h
#define TextToRaw_h

// -*- C++ -*-
//
// Package:    TextToRaw
// Class:      TextToRaw
// 
/**\class TextToRaw TextToRaw.cc L1Triggr/TextToDigi/src/TextToRaw.cc

 Description: Convert ASCII dump of a raw event to FEDRawData format for unpacking

 Implementation:
     Input format is a 32 bit hex string per line (LSW first). Events separated with blank line.
*/
//
// Original Author:  Jim Brooke
//         Created:  Wed Nov  1 11:57:10 CET 2006
// $Id: TextToRaw.h,v 1.7 2009/12/03 18:35:58 nuno Exp $
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

class TextToRaw : public edm::EDProducer {
 public:
  explicit TextToRaw(const edm::ParameterSet&);
  ~TextToRaw();
  
 private: // methods
  virtual void beginJob();
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

 private:

  // ID of the FED to emulate
  int fedId_;

  // File to read
  std::string filename_;
  std::ifstream file_;

  // array to store the data
  static const unsigned EVT_MAX_SIZE=8192;
  char data_[EVT_MAX_SIZE];

  int fileEventOffset_;
  int nevt_;
  void putEmptyDigi(edm::Event&);

};

#endif
