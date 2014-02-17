#ifndef SOURCECARDTEXTTORCTDIGI_H
#define SOURCECARDTEXTTORCTDIGI_H

// -*- C++ -*-
//
// Package:    SourceCardTextToRctDigi
// Class:      SourceCardTextToRctDigi
// 
/**\class SourceCardTextToRctDigi SourceCardTextToRctDigi.h L1Trigger/TextToDigi/src/SourceCardTextToRctDigi.h

 Description: Input text file to be loaded into the source cards and output RCT digis for pattern tests. 

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Alex Tapper
//         Created:  Fri Mar  9 19:11:51 CET 2007
// $Id: SourceCardTextToRctDigi.h,v 1.2 2007/05/09 11:21:18 tapper Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

// RCT data includes
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"

#include "L1Trigger/TextToDigi/src/SourceCardRouting.h"

#include <iostream>
#include <fstream>

class SourceCardTextToRctDigi : public edm::EDProducer {
 public:
  explicit SourceCardTextToRctDigi(const edm::ParameterSet&);
  ~SourceCardTextToRctDigi();
  
 private:
  virtual void produce(edm::Event&, const edm::EventSetup&);

  /// Create empty digi collection
  void putEmptyDigi(edm::Event&);

  /// Name out input file
  std::string m_textFileName;

  /// Number of events to skip at the start of the file
  int m_fileEventOffset;

  /// Event counter
  int m_nevt;

  /// file handle
  std::ifstream m_file;
  
  /// source card router
  SourceCardRouting m_scRouting;
  
};

#endif
