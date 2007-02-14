
#ifndef GCTBLOCKCONVERTER_H
#define GCTBLOCKCONVERTER_H

#include <vector>
#include <map>
#include <memory>

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
  void convertBlock(const unsigned char * d, unsigned id);

  // set collection pointers
  void setEmCollection(L1GctEmCandCollection* coll) { gctEm = coll; }
  void setInterEmCollection(L1GctEmCandCollection* coll) { gctInterEm = coll; }

 private:


 private:

  // block info
  std::map<unsigned, unsigned> blockLength_;  // size of a block

  // collections of output objects
  L1GctEmCandCollection* gctEm;
  L1GctEmCandCollection* gctInterEm;  

};

#endif
