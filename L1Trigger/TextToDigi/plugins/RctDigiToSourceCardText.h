#ifndef RCTDIGITOSOURCECARDTEXT_H
#define RCTDIGITOSOURCECARDTEXT_H

// -*- C++ -*-
//
// Package:    RctDigiToSourceCardText
// Class:      RctDigiToSourceCardText
//
/**\class RctDigiToSourceCardText RctDigiToSourceCardText.h
 L1Trigger/TextToDigi/interface/RctDigiToSourceCardText.h

 Description: Input RCT digis and output text file to be loaded into the source
 cards for pattern tests.

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Alex Tapper
//         Created:  Fri Feb 16 14:52:19 CET 2007
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

// RCT data includes
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"

#include "L1Trigger/TextToDigi/interface/SourceCardRouting.h"

#include <fstream>
#include <iostream>

class RctDigiToSourceCardText : public edm::one::EDAnalyzer<> {
public:
  explicit RctDigiToSourceCardText(const edm::ParameterSet &);
  ~RctDigiToSourceCardText() override;

private:
  void analyze(const edm::Event &, const edm::EventSetup &) override;

  /// Label for RCT digis
  edm::InputTag m_rctInputLabel;

  /// Name out output file
  std::string m_textFileName;

  /// file handle
  std::ofstream m_file;

  /// event counter
  unsigned short m_nevt;

  /// source card router
  SourceCardRouting m_scRouting;
};

#endif
