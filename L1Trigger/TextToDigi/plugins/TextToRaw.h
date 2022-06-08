#ifndef TextToRaw_h
#define TextToRaw_h

// -*- C++ -*-
//
// Package:    TextToRaw
// Class:      TextToRaw
//
/**\class TextToRaw TextToRaw.cc L1Triggr/TextToDigi/src/TextToRaw.cc

 Description: Convert ASCII dump of a raw event to FEDRawData format for
 unpacking

 Implementation:
     Input format is a 32 bit hex string per line (LSW first). Events separated
 with blank line.
*/
//
// Original Author:  Jim Brooke
//         Created:  Wed Nov  1 11:57:10 CET 2006
//
//

// system include files
#include <fstream>
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"

//
// class decleration
//

class TextToRaw : public edm::one::EDProducer<> {
public:
  explicit TextToRaw(const edm::ParameterSet &);
  ~TextToRaw() override;

private:  // methods
  void beginJob() override;
  void produce(edm::Event &, const edm::EventSetup &) override;
  void endJob() override;

private:
  // ID of the FED to emulate
  int fedId_;

  // File to read
  std::string filename_;
  std::ifstream file_;

  // array to store the data
  static constexpr unsigned EVT_MAX_SIZE = 8192;
  char data_[EVT_MAX_SIZE];

  int fileEventOffset_;
  int nevt_;
  void putEmptyDigi(edm::Event &);
};

#endif
