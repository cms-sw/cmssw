
#ifndef GCTBLOCKCONVERTER_H
#define GCTBLOCKCONVERTER_H

#include <vector>
#include <map>
#include <memory>
#include <boost/cstdint.hpp>

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"



class GctBlockConverter {
 public:

  GctBlockConverter();
  ~GctBlockConverter();
  
  /// find out whether a block ID is recognised
  bool validBlock(unsigned id);

  /// return block length in 32-bit words
  unsigned blockLength(unsigned id);
    
  /// set collection pointers
  /// when unpacking set these to empty collections which will be filled
  /// when packing, set these to the full collections of input data
  void setRctEmCollection(L1CaloEmCollection* coll) { rctEm_ = coll; }
  void setIsoEmCollection(L1GctEmCandCollection* coll) { gctIsoEm_ = coll; }
  void setNonIsoEmCollection(L1GctEmCandCollection* coll) { gctNonIsoEm_ = coll; }
  void setInternEmCollection(L1GctInternEmCandCollection* coll) { gctInternEm_ = coll; }

  // get digis from block
  void convertBlock(const unsigned char * d, unsigned id, unsigned nSamples);

  /// get block from digis
  void writeBlock(unsigned char * d, unsigned id);

 private:

  // convert functions for each type of block
  /// unpack GCT EM Candidates
  void blockToGctEmCand(const unsigned char * d, unsigned id, unsigned nSamples);

  /// unpack GCT internal EM Candidates
  void blockToGctInternEmCand(const unsigned char * d, unsigned id, unsigned nSamples);

  /// unpack RCT EM Candidates
  void blockToRctEmCand(const unsigned char * d, unsigned id, unsigned nSamples);


  // reverse functions for each type of block
  /// write header for packing
  void writeGctHeader(unsigned char * d, unsigned id);

  /// pack GCT EM Candidates (for triggered crossing only)
  void gctEmCandToBlock(unsigned char * d, unsigned id);

  /// pack RCT EM Candidates (for triggered crossing only)
  void rctEmCandToBlock(unsigned char * d, unsigned id);


 private:

  // block info
  std::map<unsigned, unsigned> blockLength_;  // size of a block

  // map of conversion functions
  typedef  void (GctBlockConverter::*convFn)(uint16_t, uint16_t, int);
  std::map< unsigned, convFn > convertFn_;
  
  // collections of RCT objects
  L1CaloEmCollection* rctEm_;

  // collections of output objects
  L1GctEmCandCollection* gctIsoEm_;
  L1GctEmCandCollection* gctNonIsoEm_;
  L1GctInternEmCandCollection* gctInternEm_;  

};

#endif
