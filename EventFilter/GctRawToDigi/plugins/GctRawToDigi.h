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
// $Id: GctRawToDigi.h,v 1.28 2009/04/09 11:10:39 frazier Exp $
//
//

// system include files
#include <memory>
#include <ostream>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"

#include "EventFilter/GctRawToDigi/src/GctUnpackCollections.h"
#include "EventFilter/GctRawToDigi/src/GctFormatTranslateBase.h"


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
  
  /// Unpacks the raw data
  /*! \param invalidDataFlag - if true, then won't attempt unpack but just output empty collecions. */
  void unpack(const FEDRawData& d, edm::Event& e, GctUnpackCollections * const colls);

  /// Looks at the firmware version header in the S-Link packet and instantiates relevant format translator.
  /*! Returns false if it fails to instantiate a Format Translator */
  bool autoDetectRequiredFormatTranslator(const unsigned char * data);

  /// Prints out a list of blocks and the various numbers of trigger objects that have been unpacked from them.
  void doVerboseOutput(const GctBlockHeaderCollection& bHdrs, const GctUnpackCollections * const colls) const;

  virtual void endJob();

private: // members

  /// The maximum number of blocks we will try to unpack before thinking something is wrong
  static const unsigned MAX_BLOCKS = 256;

  edm::InputTag inputLabel_;  ///< FED collection label.
  int fedId_;                 ///< GCT FED ID.

  // unpacking options
  const bool hltMode_;  ///< If true, only outputs the GT output data, and only BX = 0.
  const unsigned formatVersion_;  ///< Defines unpacker verison to be used (e.g.: "Auto-detect", "MCLegacy", "V35", etc).
  const bool verbose_;  ///< If true, then debug print out for each event.

  // Block to Digi converter
  GctFormatTranslateBase * formatTranslator_;

  unsigned unpackFailures_;  ///< To count the total number of GCT unpack failures.  
};

#endif
