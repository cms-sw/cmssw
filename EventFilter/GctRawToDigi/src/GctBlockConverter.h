
#ifndef GCTBLOCKCONVERTER_H
#define GCTBLOCKCONVERTER_H

#include <vector>
#include <map>
#include <memory>
#include <boost/cstdint.hpp>

#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"



class GctBlockConverter {
 public:

  GctBlockConverter();
  ~GctBlockConverter();
  
  // recognised block ID
  bool validBlock(unsigned id);

  // return block length in 32-bit words
  unsigned blockLength(unsigned id);
    
  // convert a block
  void convertBlock(const unsigned char * d, unsigned id, unsigned nSamples);

  // set collection pointers
  void setRctEmCollection(L1CaloEmCollection* coll) { rctEm_ = coll; }
  void setIsoEmCollection(L1GctEmCandCollection* coll) { gctIsoEm_ = coll; }
  void setNonIsoEmCollection(L1GctEmCandCollection* coll) { gctNonIsoEm_ = coll; }
  void setInternEmCollection(L1GctInternEmCandCollection* coll) { gctInternEm_ = coll; }

 private:

  // convert functions for each type of block
  void blockToRctEmCand(const unsigned char * d, unsigned id, unsigned nSamples);
  void blockToGctInternEmCand(const unsigned char * d, unsigned id, unsigned nSamples);
  void blockToGctEmCand(const unsigned char * d, unsigned id, unsigned nSamples);


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
