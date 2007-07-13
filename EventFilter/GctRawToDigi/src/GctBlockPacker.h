
#ifndef GCTBLOCKPACKER_H
#define GCTBLOCKPACKER_H

#include <vector>
#include <map>
#include <memory>
#include <boost/cstdint.hpp>

//#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtSums.h"

class GctBlockPacker {
 public:

  GctBlockPacker();
  ~GctBlockPacker();

  uint32_t bcId() { return bcid_; }
  uint32_t evId() { return evid_; }
  
  void setBcId(uint32_t ev) { evid_ = ev; }
  void setEvId(uint32_t bc) { bcid_ = bc; }

  void writeFedHeader(unsigned char * d, uint32_t fedId);

  void writeFedFooter(unsigned char * d, const unsigned char * start);

  void writeGctHeader(unsigned char * d, uint16_t id, uint16_t nSamples);

  void writeGctEmBlock(unsigned char * d, const L1GctEmCandCollection* iso, const L1GctEmCandCollection* nonIso);

  void writeGctCenJetBlock(unsigned char * d, const L1GctJetCandCollection* coll);

  void writeGctTauJetBlock(unsigned char * d, const L1GctJetCandCollection* coll);

  void writeGctForJetBlock(unsigned char * d, const L1GctJetCandCollection* coll);

  void writeEnergySumsBlock(unsigned char * d, const L1GctEtMiss* etm, const L1GctEtTotal* ett, const L1GctEtHad* eth);

 private:

  uint32_t bcid_;
  uint32_t evid_;
  
};

#endif
