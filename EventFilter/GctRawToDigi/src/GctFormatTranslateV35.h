#ifndef GctFormatTranslateV35_h_
#define GctFormatTranslateV35_h_

#include "EventFilter/GctRawToDigi/src/GctFormatTranslateBase.h"

/*!
* \class GctFormatTranslateV35
* \brief Unpacks/packs the V35 raw format
*
* \author Robert Frazier
*/

// ************************************************************************
// ***  THE TRANSLATION PROCESS MUST NEVER THROW ANY KIND OF EXCEPTION! ***
// ************************************************************************

class GctFormatTranslateV35 : public GctFormatTranslateBase {
public:
  /* PUBLIC METHODS */

  /// Constructor.
  /*! \param hltMode - set true to unpack only BX zero and GCT output data (i.e. to run as quick as possible).
   *  \param unpackSharedRegions - this is a commissioning option to unpack the shared RCT calo regions. */
  explicit GctFormatTranslateV35(bool hltMode = false, bool unpackSharedRegions = false);

  ~GctFormatTranslateV35() override;  ///< Destructor.

  /// Generate a block header from four 8-bit values.
  GctBlockHeader generateBlockHeader(const unsigned char* data) const override;

  /// Get digis from the block - will return true if it succeeds, false otherwise.
  bool convertBlock(const unsigned char* d, const GctBlockHeader& hdr) override;

  /* ------------------------------ */
  /* Public Block Packing Functions */
  /* ------------------------------ */

  // -- TO DO --

protected:
  /* PROTECTED METHODS */

  /* Static data member access methods */
  const BlockLengthMap& blockLengthMap() const final {
    return m_blockLength;
  }  ///< get the static block ID to block-length map.

  const BlockNameMap& blockNameMap() const final { return m_blockName; }  ///< get the static block ID to blockname map.

  const BlkToRctCrateMap& rctEmCrateMap() const final {
    return m_rctEmCrate;
  }  ///< get static the block ID to RCT crate map for electrons.

  const BlkToRctCrateMap& rctJetCrateMap() const final {
    return m_rctJetCrate;
  }  ///< get the static block ID to RCT crate map for jets

  const BlockIdToEmCandIsoBoundMap& internEmIsoBounds() const final {
    return m_internEmIsoBounds;
  }  ///< get the static intern EM cand isolated boundary map.

  /* Other general methods */
  /// Returns a raw 32-bit header word generated from the blockId, number of time samples, bunch-crossing and event IDs.
  uint32_t generateRawHeader(const uint32_t blockId,
                             const uint32_t nSamples,
                             const uint32_t bxId,
                             const uint32_t eventId) const override;

private:
  /* PRIVATE TYPEDEFS */

  /// Function pointer typdef to a block unpack function.
  typedef void (GctFormatTranslateV35::*PtrToUnpackFn)(const unsigned char*, const GctBlockHeader&);
  /// Typedef for a block ID to unpack function map.
  typedef std::map<unsigned int, PtrToUnpackFn> BlockIdToUnpackFnMap;

  /* PRIVATE MEMBER DATA */

  /// Map to translate block number to fundamental size of a block (i.e. for 1 time-sample).
  static const BlockLengthMap m_blockLength;

  /// Map to hold a description for each block number.
  static const BlockNameMap m_blockName;

  /// Map to relate capture block ID to the RCT crate the data originated from (for electrons).
  static const BlkToRctCrateMap m_rctEmCrate;

  /// Map to relate capture block ID to the RCT crate the data originated from (for jets).
  static const BlkToRctCrateMap m_rctJetCrate;

  /*! A map of Block IDs to IsoBoundaryPairs for storing the location of the isolated
   *  Internal EM cands in the pipeline, as this differs with Block ID. */
  static const BlockIdToEmCandIsoBoundMap m_internEmIsoBounds;

  /// Block ID to unpack function map.
  static const BlockIdToUnpackFnMap m_blockUnpackFn;

  /* PRIVATE METHODS */

  /* --------------------------------- */
  /* Private Block Unpacking Functions */
  /* --------------------------------- */

  /// unpack GCT EM Candidates and energy sums.
  void blockToGctEmCandsAndEnergySums(const unsigned char* d, const GctBlockHeader& hdr);

  /// Unpack GCT Jet Candidates and jet counts.
  void blockToGctJetCandsAndCounts(const unsigned char* d, const GctBlockHeader& hdr);

  /// unpack GCT internal EM Candidates
  void blockToGctInternEmCand(const unsigned char* d, const GctBlockHeader& hdr);

  /// unpack RCT EM Candidates
  void blockToRctEmCand(const unsigned char* d, const GctBlockHeader& hdr);

  /// Unpack RCT Calo Regions
  void blockToRctCaloRegions(const unsigned char* d, const GctBlockHeader& hdr);

  /// unpack Fibres
  void blockToFibres(const unsigned char* d, const GctBlockHeader& hdr);

  /// unpack Fibres and RCT EM Candidates
  void blockToFibresAndToRctEmCand(const unsigned char* d, const GctBlockHeader& hdr);

  /// unpack GCT internal Et sums
  void blockToGctInternEtSums(const unsigned char* d, const GctBlockHeader& hdr);

  /// unpack GCT internal output of leaf jet finder
  void blockToGctInternEtSumsAndJetCluster(const unsigned char* d, const GctBlockHeader& hdr);

  /// unpack GCT internal wheel and conc jets
  void blockToGctTrigObjects(const unsigned char* d, const GctBlockHeader& hdr);

  /// unpack GCT internal input to wheel jet sort
  void blockToGctJetClusterMinimal(const unsigned char* d, const GctBlockHeader& hdr);

  /// unpack GCT internal shared jet finder info
  void blockToGctJetPreCluster(const unsigned char* d, const GctBlockHeader& hdr);

  /// unpack GCT internal HF ring sums
  void blockToGctInternRingSums(const unsigned char* d, const GctBlockHeader& hdr);

  /// unpack GCT internal output of wheel
  void blockToGctWheelOutputInternEtAndRingSums(const unsigned char* d, const GctBlockHeader& hdr);

  /// unpack GCT internal input to wheel
  void blockToGctWheelInputInternEtAndRingSums(const unsigned char* d, const GctBlockHeader& hdr);
};

#endif
