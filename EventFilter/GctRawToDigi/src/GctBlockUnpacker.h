
#ifndef GCTBLOCKUNPACKER_H
#define GCTBLOCKUNPACKER_H

#include <vector>
#include <map>
#include <memory>
#include <boost/cstdint.hpp>

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"

#include "L1Trigger/TextToDigi/src/SourceCardRouting.h"

#include "EventFilter/GctRawToDigi/src/GctBlockHeader.h"

class GctBlockUnpacker {
 public:

  GctBlockUnpacker();
  ~GctBlockUnpacker();
  
  /// set collection pointers
  /// when unpacking set these to empty collections which will be filled
  /// when packing, set these to the full collections of input data
  void setRctEmCollection(L1CaloEmCollection* coll) { rctEm_ = coll; }
  void setIsoEmCollection(L1GctEmCandCollection* coll) { gctIsoEm_ = coll; }
  void setNonIsoEmCollection(L1GctEmCandCollection* coll) { gctNonIsoEm_ = coll; }
  void setInternEmCollection(L1GctInternEmCandCollection* coll) { gctInternEm_ = coll; }
  void setFibreCollection(L1GctFibreCollection* coll) { gctFibres_ = coll; }

  // get digis from block
  void convertBlock(const unsigned char * d, GctBlockHeader& hdr);

 private:

  // source card mapping info
  SourceCardRouting srcCardRouting_;

  typedef std::map<unsigned int, unsigned int> RctCrateMap; // RCT Crate Map typedef.
  static RctCrateMap rctCrate_;  // And the RCT Crate Map itself.

  // Typedefs for the block ID to unpack function mapping.
  typedef void (GctBlockUnpacker::*PtrToUnpackFn)(const unsigned char *, const GctBlockHeader&);
  typedef std::map<unsigned int, PtrToUnpackFn> BlockIdToUnpackFnMap;
  
  static BlockIdToUnpackFnMap blockUnpackFn_;  // And the block ID to unpack function map itself.

  // collections of RCT objects
  L1CaloEmCollection* rctEm_;

  // collections of output objects
  L1GctEmCandCollection* gctIsoEm_;
  L1GctEmCandCollection* gctNonIsoEm_;
  L1GctInternEmCandCollection* gctInternEm_;  
  L1GctFibreCollection* gctFibres_;

 private:  // FUNCTIONS

  // convert functions for each type of block
  /// unpack GCT EM Candidates
  void blockToGctEmCand(const unsigned char * d, const GctBlockHeader& hdr);

  /// unpack GCT internal EM Candidates
  void blockToGctInternEmCand(const unsigned char * d, const GctBlockHeader& hdr);

  /// unpack RCT EM Candidates
  void blockToRctEmCand(const unsigned char * d, const GctBlockHeader& hdr);

  /// unpack Fibres
  void blockToFibres(const unsigned char * d, const GctBlockHeader& hdr);
  
  /// unpack Fibres and RCT EM Candidates
  void blockToFibresAndToRctEmCand(const unsigned char * d, const GctBlockHeader& hdr);
  
  /// Do nothing
  void blockDoNothing(const unsigned char * d, const GctBlockHeader& hdr) {}

};

#endif

