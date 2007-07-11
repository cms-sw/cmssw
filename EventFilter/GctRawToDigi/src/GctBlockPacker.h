
#ifndef GCTBLOCKPACKER_H
#define GCTBLOCKPACKER_H

#include <vector>
#include <map>
#include <memory>
#include <boost/cstdint.hpp>

//#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"

class GctBlockPacker {
 public:

  GctBlockPacker();
  ~GctBlockPacker();
  

  void writeGctEmBlock(unsigned char * d, L1GctEmCandCollection* coll);

  void writeGctCenJetBlock(unsigned char * d, L1GctJetCandCollection* coll);

  void writeGctTauJetBlock(unsigned char * d, L1GctJetCandCollection* coll);

  void writeGctForJetBlock(unsigned char * d, L1GctJetCandCollection* coll);

 private:

  // reverse functions for each type of block
  /// write header for packing
  void writeGctHeader(unsigned char * d, unsigned id);

};

#endif
