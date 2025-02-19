#ifndef GctFormatTranslateV38_h_
#define GctFormatTranslateV38_h_

#include "EventFilter/GctRawToDigi/src/GctFormatTranslateBase.h"

/*!
* \class GctFormatTranslateV38
* \brief Unpacks/packs the V38 raw format
*
* \author Robert Frazier
* $Revision: 1.4 $
* $Date: 2009/11/16 20:57:13 $
*/ 

// ************************************************************************
// ***  THE TRANSLATION PROCESS MUST NEVER THROW ANY KIND OF EXCEPTION! *** 
// ************************************************************************

class GctFormatTranslateV38 : public GctFormatTranslateBase
{
public:

  /* PUBLIC METHODS */

  /// Constructor.
  /*! \param hltMode - set true to unpack only BX zero and GCT output data (i.e. to run as quick as possible).
   *  \param unpackSharedRegions - this is a commissioning option to unpack the shared RCT calo regions. */
  explicit GctFormatTranslateV38(bool hltMode = false, 
                                 bool unpackSharedRegions = false, 
                                 unsigned numberOfGctSamplesToUnpack=1,
                                 unsigned numberOfRctSamplesToUnpack=1);
  
  virtual ~GctFormatTranslateV38(); ///< Destructor.

  /// Generate a block header from four 8-bit values.
  virtual GctBlockHeader generateBlockHeader(const unsigned char * data) const;
  
  /// Get digis from the block - will return true if it succeeds, false otherwise.
  virtual bool convertBlock(const unsigned char * d, const GctBlockHeader& hdr);


  /* ------------------------------ */
  /* Public Block Packing Functions */
  /* ------------------------------ */

  // -- TO DO --


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

  /* PRIVATE TYPEDEFS */
 
  /// Function pointer typdef to a block unpack function.
  typedef void (GctFormatTranslateV38::*PtrToUnpackFn)(const unsigned char *, const GctBlockHeader&);
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
  
  /// Number of BXs of GCT data to unpack (assuming they are in the raw data)
  const unsigned m_numberOfGctSamplesToUnpack; 

  ///< Number of BXs of RCT data to unpack (assuming they are in the raw data)
  const unsigned m_numberOfRctSamplesToUnpack; 

  /* PRIVATE METHODS */

  /* --------------------------------- */
  /* Private Block Unpacking Functions */
  /* --------------------------------- */
  
  /// unpack GCT EM Candidates and energy sums.
  void blockToGctEmCandsAndEnergySums(const unsigned char * d, const GctBlockHeader& hdr);
  
  /// Unpack GCT Jet Candidates and jet counts.
  void blockToGctJetCandsAndCounts(const unsigned char * d, const GctBlockHeader& hdr);

  /// unpack GCT internal EM Candidates
  void blockToGctInternEmCand(const unsigned char * d, const GctBlockHeader& hdr);

  /// unpack RCT EM Candidates
  void blockToRctEmCand(const unsigned char * d, const GctBlockHeader& hdr);

  /// Unpack RCT Calo Regions
  void blockToRctCaloRegions(const unsigned char * d, const GctBlockHeader& hdr);

  /// unpack Fibres
  void blockToFibres(const unsigned char * d, const GctBlockHeader& hdr);
  
  /// unpack Fibres and RCT EM Candidates
  void blockToFibresAndToRctEmCand(const unsigned char * d, const GctBlockHeader& hdr);

  /// unpack GCT internal Et sums
  void blockToGctInternEtSums(const unsigned char * d, const GctBlockHeader& hdr);

  /// unpack GCT internal output of leaf jet finder
  void blockToGctInternEtSumsAndJetCluster(const unsigned char * d, const GctBlockHeader& hdr);

  /// unpack GCT internal wheel and conc jets
  void blockToGctTrigObjects(const unsigned char * d, const GctBlockHeader& hdr);

  /// unpack GCT internal input to wheel jet sort
  void blockToGctJetClusterMinimal(const unsigned char * d, const GctBlockHeader& hdr);

  /// unpack GCT internal shared jet finder info
  void blockToGctJetPreCluster(const unsigned char * d, const GctBlockHeader& hdr);

  /// unpack GCT internal HF ring sums
  void blockToGctInternRingSums(const unsigned char * d, const GctBlockHeader& hdr);

  /// unpack GCT internal output of wheel
  void blockToGctWheelOutputInternEtAndRingSums(const unsigned char * d, const GctBlockHeader& hdr);

  /// unpack GCT internal input to wheel
  void blockToGctWheelInputInternEtAndRingSums(const unsigned char * d, const GctBlockHeader& hdr);

  /// unpack GCT internal Missing Ht data that is being input to the wheels.
  void blockToGctInternHtMissPreWheel(const unsigned char* d, const GctBlockHeader& hdr); 

  /// unpack GCT internal Missing Ht data that is either wheel output or concJet input (i.e. after wheel processing).
  void blockToGctInternHtMissPostWheel(const unsigned char* d, const GctBlockHeader& hdr);
};

#endif
