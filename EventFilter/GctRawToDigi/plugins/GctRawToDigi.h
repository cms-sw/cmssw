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
// $Id: GctRawToDigi.h,v 1.17 2008/03/19 18:24:22 frazier Exp $
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

#include "EventFilter/GctRawToDigi/src/GctBlockUnpackerBase.h"


// *******************************************************************
// ***  THE UNPACK PROCESS MUST NEVER THROW ANY KIND OF EXCEPTION! *** 
// *******************************************************************

class GctRawToDigi : public edm::EDProducer
{
public:

  explicit GctRawToDigi(const edm::ParameterSet&);
  ~GctRawToDigi();

  
private: // methods

  virtual void beginJob(const edm::EventSetup&) ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  /// Unpacks the raw data
  /*! \param invalidDataFlag - if true, then won't attempt unpack but just output empty collecions. */
  void unpack(const FEDRawData& d, edm::Event& e, const bool invalidDataFlag=false);


private: // members

  /// The maximum number of blocks we will try to unpack before thinking something is wrong
  static const unsigned MAX_BLOCKS = 256;

  edm::InputTag inputLabel_;  // FED collection label.
  int fedId_;                 // GCT FED ID.
  const bool verbose_;        // If true, then debug print out for each event.

  // unpacking options
  const bool hltMode_;  // If true, only outputs the GT output data, and only BX = 0.
  const bool grenCompatibilityMode_;  // If true, use old-style (GREN07 era) block headers & pipe format.
  const bool doEm_;
  const bool doJets_;
  const bool doEtSums_;
  const bool doInternEm_;
  const bool doRct_;
  const bool doFibres_;

  // Block to Digi converter
  GctBlockUnpackerBase * blockUnpacker_;

  unsigned unpackFailures_;
};

#endif
