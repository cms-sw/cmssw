
#ifndef GCTBLOCKPACKER_H
#define GCTBLOCKPACKER_H

// C++ headers
#include <vector>
#include <map>
#include <memory>
#include <boost/cstdint.hpp>

// CMSSW DataFormats headers
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtSums.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCounts.h"


class GctBlockPacker
{
 public:

  GctBlockPacker();
  ~GctBlockPacker();

  uint32_t bcId() { return bcid_; }
  uint32_t evId() { return evid_; }
  
  void setBcId(uint32_t ev) { evid_ = ev; }
  void setEvId(uint32_t bc) { bcid_ = bc; }

  void writeGctHeader(unsigned char * d, uint16_t id, uint16_t nSamples);

  /// Writes GCT output jet cands and counts into an unsigned char array, starting at the position pointed to by d.
  /*! \param d must be pointing at the position where the Jet Output block header should be written! */
  void writeGctOutJetBlock(unsigned char * d, const L1GctJetCandCollection* cenJets,
                           const L1GctJetCandCollection* forJets, const L1GctJetCandCollection* tauJets, 
                           const L1GctJetCounts* jetCounts);
  
  /// Writes GCT output EM and energy sums block into an unsigned char array, starting at the position pointed to by d.
  /*! \param d must be pointing at the position where the EM Output block header should be written! */
  void writeGctOutEmAndEnergyBlock(unsigned char * d, const L1GctEmCandCollection* iso,
                                   const L1GctEmCandCollection* nonIso, const L1GctEtTotal* etTotal,
                                   const L1GctEtHad* etHad, const L1GctEtMiss* etMiss);

 private:

  /// An enum for use with central, forward, and tau jet cand collections vector(s).
  /*! Note that the order here mimicks the order in the RAW data format. */
  enum JetCandCatagory { TAU_JETS, FORWARD_JETS, CENTRAL_JETS, NUM_JET_CATAGORIES };

  uint32_t bcid_;
  uint32_t evid_;
  
};

#endif
