
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

  void writeGctHeader(unsigned char * d, uint16_t id, uint16_t nSamples);

  /// Writes the GCT EM block into an unsigned char array, starting at the position pointed to by d.
  /*! \param d must be pointing at the position where the EM block header should be written! */
  void writeGctEmBlock(unsigned char * d, const L1GctEmCandCollection* iso, const L1GctEmCandCollection* nonIso);

  /// Writes the GCT Jet block into an unsigned char array, starting at the position pointed to by d.
  /*! \param d must be pointing at the position where the Jet block header should be written! */
  void writeGctJetBlock(unsigned char * d, const L1GctJetCandCollection* cenJets,
                        const L1GctJetCandCollection* forJets, const L1GctJetCandCollection* tauJets);

  void writeEnergySumsBlock(unsigned char * d, const L1GctEtMiss* etm, const L1GctEtTotal* ett, const L1GctEtHad* eth);

 private:

  /// An enum for use with central, forward, and tau jet cand collections vector(s).
  /*! Note that the order here mimicks the order in the RAW data format. */
  enum JetCandCatagory { TAU_JETS, FORWARD_JETS, CENTRAL_JETS, NUM_JET_CATAGORIES };

  uint32_t bcid_;
  uint32_t evid_;
  
};

#endif
