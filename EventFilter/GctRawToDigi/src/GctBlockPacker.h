
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

// Sourcecard routing
#include "L1Trigger/TextToDigi/src/SourceCardRouting.h"


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
                           const L1GctJetCountsCollection* jetCounts);
  
  /// Writes GCT output EM and energy sums block into an unsigned char array, starting at the position pointed to by d.
  /*! \param d must be pointing at the position where the EM Output block header should be written! */
  void writeGctOutEmAndEnergyBlock(unsigned char * d, const L1GctEmCandCollection* iso,
                                   const L1GctEmCandCollection* nonIso, const L1GctEtTotalCollection* etTotal,
                                   const L1GctEtHadCollection* etHad, const L1GctEtMissCollection* etMiss);

  /// Writes the 4 RCT EM Candidate blocks.
  void writeRctEmCandBlocks(unsigned char * d, const L1CaloEmCollection * rctEm);

  /// Writes the giant hack that is the RCT Calo Regions block.
  void writeRctCaloRegionBlock(unsigned char * d, const L1CaloRegionCollection * rctCalo);

 private:

  /// An enum for use with central, forward, and tau jet cand collections vector(s).
  /*! Note that the order here mimicks the order in the RAW data format. */
  enum JetCandCatagory { TAU_JETS, FORWARD_JETS, CENTRAL_JETS, NUM_JET_CATEGORIES };

  /// Typedef for mapping block ID to the first RCT crate in that block
  typedef std::map<unsigned int, unsigned int> RctCrateMap;
  
  uint32_t bcid_;
  uint32_t evid_;
  
  /// Source card mapping info
  SourceCardRouting srcCardRouting_;

  /// Map to relate capture block ID to the RCT crate the data originated from.
  static RctCrateMap rctCrate_;
  
  /// Struct of all data needed for running the emulator to SFP (sourcecard optical output) conversion.
  struct EmuToSfpData
  {
    // Input data.
    unsigned short eIsoRank[4];
    unsigned short eIsoCardId[4];
    unsigned short eIsoRegionId[4];
    unsigned short eNonIsoRank[4];
    unsigned short eNonIsoCardId[4];
    unsigned short eNonIsoRegionId[4];
    unsigned short mipBits[7][2];
    unsigned short qBits[7][2];
    // Output data.
    unsigned short sfp[2][4]; // [ cycle ] [ output number ]
  };

};

#endif
