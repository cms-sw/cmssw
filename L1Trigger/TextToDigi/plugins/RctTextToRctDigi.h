#ifndef RCTTEXTTORCTDIGI_H
#define RCTTEXTTORCTDIGI_H

// -*- C++ -*-
//
// Package:    RctTextToRctDigi
// Class:      RctTextToRctDigi
//
/**\class RctTextToRctDigi RctTextToRctDigi.h
 L1Trigger/TextToDigi/src/RctTextToRctDigi.h

 Description: Makes RCT digis from the file format specified by Pam Klabbers

*/
//
// Original Author:  Alex Tapper
//         Created:  Fri Mar  9 19:11:51 CET 2007
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

// RCT data includes
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"

#include <fstream>
#include <iostream>

class RctTextToRctDigi : public edm::one::EDProducer<> {
public:
  explicit RctTextToRctDigi(const edm::ParameterSet &);
  ~RctTextToRctDigi() override;

private:
  void produce(edm::Event &, const edm::EventSetup &) override;

  /// Create empty digi collection
  void putEmptyDigi(edm::Event &);

  /// Synchronize bunch crossing
  void bxSynchro(int &, int);

  /// Name out input file
  std::string m_textFileName;

  /// Number of events to be offset wrt input
  int m_fileEventOffset;

  /// Event counter
  int m_nevt;

  /// file handle
  std::ifstream m_file[18];
};

#endif
