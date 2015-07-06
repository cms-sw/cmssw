#ifndef RctRawToDigi_h
#define RctRawToDigi_h

// -*- C++ -*-
//
// Package:    RctRawToDigi
// Class:      RctRawToDigi
// 
/**\class RctRawToDigi RctRawToDigi.cc EventFilter/RctRawToDigi/src/RctRawToDigi.cc

 Description: Produce RCT digis from raw data

 Implementation:
     <Notes on implementation>
*/

// system include files
#include <memory>
#include <ostream>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "EventFilter/RctRawToDigi/src/RctUnpackCollections.h"
#include "EventFilter/RctRawToDigi/src/RCTInfo.hh"
#include "EventFilter/RctRawToDigi/src/CTP7Format.hh"
#include "EventFilter/L1TRawToDigi/interface/AMCSpec.h"
#include "EventFilter/RctRawToDigi/src/RctDataDecoder.hh"

// *******************************************************************
// ***  THE UNPACK PROCESS MUST NEVER THROW ANY KIND OF EXCEPTION! *** 
// *******************************************************************

class RctRawToDigi : public edm::stream::EDProducer<>
{
public:

  explicit RctRawToDigi(const edm::ParameterSet&);
  ~RctRawToDigi();

  //do we need this?
  //static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
 
private: // methods

  void produce(edm::Event&, const edm::EventSetup&);
  
  /// Unpacks the raw data
  /*! \param invalidDataFlag - if true, then won't attempt unpack but just output empty collecions. */
  void unpack(const FEDRawData& d, edm::Event& e, RctUnpackCollections * const colls);

  void unpackCTP7(const uint32_t *data, const unsigned block_id, const unsigned size, RctUnpackCollections * const colls);

  bool decodeLinkID(const uint32_t inputValue, uint32_t &crateNumber, uint32_t &linkNumber, bool &even);

  bool printAll(const unsigned char *data, const unsigned size);
  /// Looks at the firmware version header in the S-Link packet and instantiates relevant format translator.
  /// check block headers for consistency
  void checkHeaders();

  /// method called at job end - use to print summary report
  virtual void endJob();


private: // members

  // SLink Header Size: 64bits
  static const unsigned sLinkHeaderSize_ = 8;

  // SLink Trailer Size: 64bits
  static const unsigned sLinkTrailerSize_ = 8;

  // SLink Header Size: two 64bit words
  static const unsigned amc13HeaderSize_ = 16;

  // SLink Trailer Size: 64bits
  static const unsigned amc13TrailerSize_ = 8;

  // CTP7 Header Size: 64bits
  static const unsigned ctp7HeaderSize_ = 8;

  // CTP7 Trailer Size: 64bits
  static const unsigned ctp7TrailerSize_ = 8;

  /// The maximum number of blocks we will try to unpack before thinking something is wrong
  static const unsigned MAX_DATA = 4680;

  /// The minimum number of blocks we will try to unpack before thinking something is wrong (really this should be 920, to be tested)
  static const unsigned MIN_DATA = 900;

  // unpacking options
  edm::InputTag inputLabel_;  ///< FED collection label.
  int fedId_;                 ///< RCT FED ID.

  const bool verbose_;       ///< If true, then debug print out for each event.

  // vector of unpacked block headers, for verbostity and/or sync checks
  //RctBlockHeaderCollection blockHeaders_;

  // error handling
  static const unsigned MAX_ERR_CODE = 6;
  L1TriggerErrorCollection * errors_;    ///< pointer to error collection
  std::vector<unsigned> errorCounters_;  ///< Counts number of errors for each code (index)
  unsigned unpackFailures_;  ///< To count the total number of RCT unpack failures.  

};

#endif
