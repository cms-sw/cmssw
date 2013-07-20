#ifndef GctFormatTranslateMCLegacy_h_
#define GctFormatTranslateMCLegacy_h_

#include "EventFilter/GctRawToDigi/src/GctFormatTranslateBase.h"

/*!
* \class GctFormatTranslateMCLegacy
* \brief Unpacks/packs the MC Legacy data originally produced by the GctBlockPacker class.
* 
* The data produced by the legacy GctBlockPacker class should have a firmware version header
* that wasn't set to anything, i.e.: 0x00000000
*  
* \author Robert Frazier
* $Revision: 1.3 $
* $Date: 2009/04/21 15:33:16 $
*/ 

// ************************************************************************
// ***  THE TRANSLATION PROCESS MUST NEVER THROW ANY KIND OF EXCEPTION! *** 
// ************************************************************************

class GctFormatTranslateMCLegacy : public GctFormatTranslateBase
{
public:

  /* PUBLIC METHODS */

  /// Constructor.
  /*! \param hltMode - set true to unpack only BX zero and GCT output data (i.e. to run as quick as possible).
   *  \param unpackSharedRegions - this is a commissioning option to unpack the shared RCT calo regions. */
  explicit GctFormatTranslateMCLegacy(bool hltMode = false, bool unpackSharedRegions = false);
  
  virtual ~GctFormatTranslateMCLegacy(); ///< Destructor.

  /// Generate a block header from four 8-bit values.
  virtual GctBlockHeader generateBlockHeader(const unsigned char * data) const;
  
  /// Get digis from the block - will return true if it succeeds, false otherwise.
  virtual bool convertBlock(const unsigned char * d, const GctBlockHeader& hdr);


  /* ------------------------------ */
  /* Public Block Packing Functions */
  /* ------------------------------ */
  
  /// Writes GCT output EM and energy sums block into an unsigned char array, starting at the position pointed to by d.
  /*! \param d must be pointing at the position where the EM Output block header should be written! */
  void writeGctOutEmAndEnergyBlock(unsigned char * d,
                                   const L1GctEmCandCollection* iso,
                                   const L1GctEmCandCollection* nonIso,
                                   const L1GctEtTotalCollection* etTotal,
                                   const L1GctEtHadCollection* etHad,
                                   const L1GctEtMissCollection* etMiss);

  /// Writes GCT output jet cands and counts into an unsigned char array, starting at the position pointed to by d.
  /*! \param d must be pointing at the position where the Jet Output block header should be written! */
  void writeGctOutJetBlock(unsigned char * d, 
                           const L1GctJetCandCollection* cenJets,
                           const L1GctJetCandCollection* forJets, 
                           const L1GctJetCandCollection* tauJets, 
                           const L1GctHFRingEtSumsCollection* hfRingSums,
                           const L1GctHFBitCountsCollection* hfBitCounts,
                           const L1GctHtMissCollection* htMiss);

  /// Writes the 4 RCT EM Candidate blocks.
  void writeRctEmCandBlocks(unsigned char * d, const L1CaloEmCollection * rctEm);

  /// Writes the giant hack that is the RCT Calo Regions block.
  void writeAllRctCaloRegionBlock(unsigned char * d, const L1CaloRegionCollection * rctCalo);


protected:
  
  /* PROTECTED METHODS */

  /* Static data member access methods */
  virtual BlockLengthMap& blockLengthMap() { return m_blockLength; } ///< get the static block ID to block-length map.
  virtual const BlockLengthMap& blockLengthMap() const { return m_blockLength; } ///< get the static block ID to block-length map.
  
  virtual BlockNameMap& blockNameMap() { return m_blockName; }  ///< get the static block ID to block-name map.
  virtual const BlockNameMap& blockNameMap() const { return m_blockName; }  ///< get the static block ID to blockname map.
  
  virtual BlkToRctCrateMap& rctEmCrateMap() { return m_rctEmCrate; }  ///< get the static block ID to RCT crate map for electrons.
  virtual const BlkToRctCrateMap& rctEmCrateMap() const { return m_rctEmCrate; }  ///< get static the block ID to RCT crate map for electrons.
 
  virtual BlkToRctCrateMap& rctJetCrateMap() { return m_rctJetCrate; }  ///< get the static block ID to RCT crate map for jets
  virtual const BlkToRctCrateMap& rctJetCrateMap() const { return m_rctJetCrate; }  ///< get the static block ID to RCT crate map for jets

  virtual BlockIdToEmCandIsoBoundMap& internEmIsoBounds() { return m_internEmIsoBounds; }  ///< get the static intern EM cand isolated boundary map.
  virtual const BlockIdToEmCandIsoBoundMap& internEmIsoBounds() const { return m_internEmIsoBounds; }  ///< get the static intern EM cand isolated boundary map.


  /* Other general methods */
  /// Returns a raw 32-bit header word generated from the blockId, number of time samples, bunch-crossing and event IDs.
  virtual uint32_t generateRawHeader(const uint32_t blockId,
                                     const uint32_t nSamples,
                                     const uint32_t bxId,
                                     const uint32_t eventId) const;


private:

  /* PRIVATE TYPES & TYPEDEFS */
 
  /// Function pointer typdef to a block unpack function.
  typedef void (GctFormatTranslateMCLegacy::*PtrToUnpackFn)(const unsigned char *, const GctBlockHeader&);
  /// Typedef for a block ID to unpack function map.
  typedef std::map<unsigned int, PtrToUnpackFn> BlockIdToUnpackFnMap;


  /* PRIVATE MEMBER DATA */
  
  /// Map to translate block number to fundamental size of a block (i.e. for 1 time-sample).
  static BlockLengthMap m_blockLength;
  
  /// Map to hold a description for each block number.
  static BlockNameMap m_blockName;
  
  /// Map to relate capture block ID to the RCT crate the data originated from (for electrons).
  static BlkToRctCrateMap m_rctEmCrate;

  /// Map to relate capture block ID to the RCT crate the data originated from (for jets).
  static BlkToRctCrateMap m_rctJetCrate;
  
  /*! A map of Block IDs to IsoBoundaryPairs for storing the location of the isolated
   *  Internal EM cands in the pipeline, as this differs with Block ID. */ 
  static BlockIdToEmCandIsoBoundMap m_internEmIsoBounds;

  /// Block ID to unpack function map.
  static BlockIdToUnpackFnMap m_blockUnpackFn;
  

  /* PRIVATE METHODS */
  
  /* --------------------------------- */
  /* Private Block Unpacking Functions */
  /* --------------------------------- */
  
  /// unpack GCT EM Candidates and energy sums.
  void blockToGctEmCandsAndEnergySums(const unsigned char * d, const GctBlockHeader& hdr);
  
  /// Unpack GCT Jet Candidates and jet counts.
  void blockToGctJetCandsAndCounts(const unsigned char * d, const GctBlockHeader& hdr);

  /// unpack RCT EM Candidates
  void blockToRctEmCand(const unsigned char * d, const GctBlockHeader& hdr);

  /// unpack Fibres
  void blockToFibres(const unsigned char * d, const GctBlockHeader& hdr);
  
  /// unpack Fibres and RCT EM Candidates
  void blockToFibresAndToRctEmCand(const unsigned char * d, const GctBlockHeader& hdr);

  /// Unpack All RCT Calo Regions ('orrible hack for DigiToRaw use)
  void blockToAllRctCaloRegions(const unsigned char * d, const GctBlockHeader& hdr);


  /* ----------------------------- */
  /* Miscellaneous Private Methods */
  /* ----------------------------- */
    
  /// Template function (used in packing) that will find the offset to first item in a collection vector where bx=0.
  /*! Returns false if fails to find any item in the collection with bx=0 */  
  template <typename Collection> 
  bool findBx0OffsetInCollection(unsigned& bx0Offset, const Collection* coll);

};

#endif
