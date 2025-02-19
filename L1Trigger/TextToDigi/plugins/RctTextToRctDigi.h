#ifndef RCTTEXTTORCTDIGI_H
#define RCTTEXTTORCTDIGI_H

// -*- C++ -*-
//
// Package:    RctTextToRctDigi
// Class:      RctTextToRctDigi
// 
/**\class RctTextToRctDigi RctTextToRctDigi.h L1Trigger/TextToDigi/src/RctTextToRctDigi.h

 Description: Makes RCT digis from the file format specified by Pam Klabbers

*/
//
// Original Author:  Alex Tapper
//         Created:  Fri Mar  9 19:11:51 CET 2007
// $Id: RctTextToRctDigi.h,v 1.3 2007/08/06 11:38:41 nuno Exp $
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

#include <iostream>
#include <fstream>

class RctTextToRctDigi : public edm::EDProducer {
 public:
  explicit RctTextToRctDigi(const edm::ParameterSet&);
  ~RctTextToRctDigi();
  
 private:
  virtual void produce(edm::Event&, const edm::EventSetup&);

  /// Create empty digi collection
  void putEmptyDigi(edm::Event&);

  /// Synchronize bunch crossing
  void bxSynchro(int &,int);

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
