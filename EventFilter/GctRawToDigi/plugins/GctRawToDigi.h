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
//
//

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

#include "EventFilter/GctRawToDigi/src/GctUnpackCollections.h"
#include "EventFilter/GctRawToDigi/src/GctFormatTranslateBase.h"


// *******************************************************************
// ***  THE UNPACK PROCESS MUST NEVER THROW ANY KIND OF EXCEPTION! *** 
// *******************************************************************

class GctRawToDigi : public edm::stream::EDProducer<>
{
public:

  explicit GctRawToDigi(const edm::ParameterSet&);
  ~GctRawToDigi();

   static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
 
private: // methods

  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  
  /// Unpacks the raw data
  /*! \param invalidDataFlag - if true, then won't attempt unpack but just output empty collecions. */
  void unpack(const FEDRawData& d, edm::Event& e, GctUnpackCollections * const colls);

  /// Looks at the firmware version header in the S-Link packet and instantiates relevant format translator.
  /*! Returns false if it fails to instantiate a Format Translator */
  bool autoDetectRequiredFormatTranslator(const unsigned char * data);

  /// check block headers for consistency
  void checkHeaders();

  /// Prints out a list of blocks and the various numbers of trigger objects that have been unpacked from them.
  void doVerboseOutput(const GctBlockHeaderCollection& bHdrs, const GctUnpackCollections * const colls) const;

  // add an error to the error collection
  void addError(const unsigned code);

  /// method called at job end - use to print summary report
  virtual void endJob();


private: // members

  /// The maximum number of blocks we will try to unpack before thinking something is wrong
  static const unsigned MAX_BLOCKS = 256;

  // unpacking options
  edm::InputTag inputLabel_;  ///< FED collection label.
  int fedId_;                 ///< GCT FED ID.

  const bool hltMode_;        ///< If true, only outputs the GCT data sent to the GT (number of BXs defined by numberOfGctSamplesToUnpack_)
  const unsigned numberOfGctSamplesToUnpack_; ///< Number of BXs of GCT data to unpack (assuming they are in the raw data)
  const unsigned numberOfRctSamplesToUnpack_; ///< Number of BXs of RCT data to unpack (assuming they are in the raw data)
  const bool unpackSharedRegions_;  ///< Commissioning option: if true, where applicable the shared RCT calo regions will also be unpacked.
  const unsigned formatVersion_;  ///< Defines unpacker verison to be used (e.g.: "Auto-detect", "MCLegacy", "V35", etc).
  const bool checkHeaders_;  ///< If true, check block headers for synchronisation
  const bool verbose_;       ///< If true, then debug print out for each event.

  // format translator
  GctFormatTranslateBase * formatTranslator_;  ///< pointer to the block-to-digi converter

  // vector of unpacked block headers, for verbostity and/or sync checks
  GctBlockHeaderCollection blockHeaders_;

  // error handling
  static const unsigned MAX_ERR_CODE = 6;
  L1TriggerErrorCollection * errors_;    ///< pointer to error collection
  std::vector<unsigned> errorCounters_;  ///< Counts number of errors for each code (index)
  unsigned unpackFailures_;  ///< To count the total number of GCT unpack failures.  

};

#endif
