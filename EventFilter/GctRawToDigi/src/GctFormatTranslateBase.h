#ifndef GctFormatTranslateBase_h_
#define GctFormatTranslateBase_h_

// C++ includes
#include <map>
#include <utility>
#include <string>

// CMSSW includes
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "L1Trigger/TextToDigi/src/SourceCardRouting.h"
#include "EventFilter/GctRawToDigi/src/GctBlockHeader.h"
#include "EventFilter/GctRawToDigi/src/GctUnpackCollections.h"


/*!
* \class GctFormatTranslateBase
* \brief Abstract interface for RawToDigi/DigiToRaw conversions of GCT data.
* 
* This class provides the common interface/functionality for the various
* concrete classes that can translate to/from specific RAW formats.
*
* \author Robert Frazier
* $Revision: 1.4 $
* $Date: 2009/09/18 15:07:06 $
*/ 


class GctFormatTranslateBase
{
public:

  /// Constructor.
  /*! \param hltMode - set true to unpack only BX zero and GCT output data (i.e. to run as quick as possible).
   *  \param unpackSharedRegions - this is a commissioning option to unpack the shared RCT calo regions. */
  explicit GctFormatTranslateBase(bool hltMode = false, bool unpackSharedRegions = false);
  
  virtual ~GctFormatTranslateBase(); ///< Destructor.
  
  /// Set the pointer to the unpack collections
  void setUnpackCollections(GctUnpackCollections * const collections) { m_collections = collections; }
  
  // Set the bunch crossing ID that will be put into the block headers when packing raw data (i.e. Digi -> Raw).
  void setPackingBxId(uint32_t bxId) { m_packingBxId = bxId; }
  
  // Set the event ID that will be put into the block headers when packing raw data (i.e. Digi -> Raw).
  void setPackingEventId(uint32_t eventId) { m_packingEventId = eventId; }

  /// Generate a block header from four 8-bit values
  virtual GctBlockHeader generateBlockHeader(const unsigned char * data) const = 0;
  
  /// Get block description
  const std::string& getBlockDescription(const GctBlockHeader& header) const;

  /// Get digis from the block - will return true if it succeeds, false otherwise.
  virtual bool convertBlock(const unsigned char * d, const GctBlockHeader& hdr) = 0;

protected:
 
  /* PROTECTED TYPDEFS, ENUMS, AND CLASS CONSTANTS. */

  /// An enum of the EM candidate types.
  /*! Note that the order here mimicks the order in the RAW data format. */
  enum EmCandCatagory { NON_ISO_EM_CANDS, ISO_EM_CANDS, NUM_EM_CAND_CATEGORIES };

  /// Useful enumeration for jet candidate pack/unpack.
  /*! Note that the order here mimicks the order in the RAW data format. */
  enum JetCandCategory { TAU_JETS, FORWARD_JETS, CENTRAL_JETS, NUM_JET_CATEGORIES };

  typedef std::map<unsigned int, unsigned int> BlockLengthMap;   ///< Block ID to Block Length map
  typedef std::pair<unsigned int, unsigned int> BlockLengthPair; ///< Block ID/length pair
  typedef std::map<unsigned int, std::string> BlockNameMap;   ///< Block ID to Block Description map
  typedef std::pair<unsigned int, std::string> BlockNamePair; ///< Block ID/Description pair

  /// Typedef for mapping block ID to RCT crate.
  typedef std::map<unsigned int, unsigned int> BlkToRctCrateMap;

  /*! A typedef that holds the inclusive lower and upper bounds of pipeline
   *  gct trigger object pair number for isolated EM candidates.
   *  I.e. if the first and second trig object pair in the pipeline payload
   *  are isolated cands (4 iso in total), then the IsoBoundaryPair would
   *  be (0,1). */ 
  typedef std::pair<unsigned int, unsigned int> IsoBoundaryPair;
  
  /// A typdef for mapping Block IDs to IsoBoundaryPairs.
  typedef std::map<unsigned int, IsoBoundaryPair> BlockIdToEmCandIsoBoundMap;


  /* PROTECTED METHODS */

  /* Static data member access methods */
  virtual BlockLengthMap& blockLengthMap() = 0; ///< get the static block ID to block-length map.
  virtual const BlockLengthMap& blockLengthMap() const = 0; ///< get the static block ID to block-length map.
  
  virtual BlockNameMap& blockNameMap() = 0;  ///< get the static block ID to block-name map.
  virtual const BlockNameMap& blockNameMap() const = 0;  ///< get the static block ID to blockname map.
  
  virtual BlkToRctCrateMap& rctEmCrateMap() = 0;  ///< get the static block ID to RCT crate map for electrons.
  virtual const BlkToRctCrateMap& rctEmCrateMap() const = 0;  ///< get static the block ID to RCT crate map for electrons.
 
  virtual BlkToRctCrateMap& rctJetCrateMap() = 0;  ///< get the static block ID to RCT crate map for jets
  virtual const BlkToRctCrateMap& rctJetCrateMap() const = 0;  ///< get the static block ID to RCT crate map for jets

  virtual BlockIdToEmCandIsoBoundMap& internEmIsoBounds() = 0;  ///< get the static intern EM cand isolated boundary map.
  virtual const BlockIdToEmCandIsoBoundMap& internEmIsoBounds() const = 0;  ///< get the static intern EM cand isolated boundary map.


  /* Data member access methods */
  GctUnpackCollections * const colls() const { return m_collections; } ///< Protected access to the GCT Unpack Collections.
  bool hltMode() const { return m_hltMode; }  ///< Protected interface to get HLT optimisation mode flag.
  bool unpackSharedRegions() const { return m_unpackSharedRegions; }  /// Protected interface to the unpackSharedRegions commissioning option.
  const SourceCardRouting& srcCardRouting() const { return m_srcCardRouting; } ///< Protected access to SourceCardRouting.
  const uint32_t packingBxId() const { return m_packingBxId; } ///< Get the BxId to be used when packing data.
  const uint32_t packingEventId() const { return m_packingEventId; } ///< Get the EventId to be used when packing data.


  /* Other general methods */
  /// Get a specific jet candandiate collection using the JetCandCategory enumeration.
  L1GctJetCandCollection * const gctJets(const unsigned cat) const;

  /// Returns a raw 32-bit header word generated from the blockId, number of time samples, bunch-crossing and event IDs.
  virtual uint32_t generateRawHeader(const uint32_t blockId,
                                     const uint32_t nSamples,
                                     const uint32_t bxId,
                                     const uint32_t eventId) const = 0;

  /// Writes a raw block header into the raw data array for a given block ID and number of time-samples.
  /*! BxId and EventId values for the raw header are set via the setPackingBxId() and setPackingEventId() methods. */
  void writeRawHeader(unsigned char * data, uint32_t blockId, uint32_t nSamples) const;

  /// Performs checks on the block header to see if the block is possible to unpack or not.
  bool checkBlock(const GctBlockHeader& hdr) const;
  
  /// The null unpack function - obviously common to all formats.
  void blockDoNothing(const unsigned char * d, const GctBlockHeader& hdr) {}
  

private:

  /* PRIVATE STATIC CONSTS */
  static const std::string INVALID_BLOCK_HEADER_STR;

  /* PRIVATE MEMBER DATA */

  /// Pointer to the output collections object.
  GctUnpackCollections * m_collections;  

  /// If true, unpack only BX zero and GCT output data (i.e. to run as quickly as possible) 
  bool m_hltMode;
  
  /// If true, the shared RCT Calo regions will be unpacked also
  /*! This is a commissioning option only - may not be relevant to all concrete implementations! */
  bool m_unpackSharedRegions;

  /// Source card mapping info
  SourceCardRouting m_srcCardRouting;
  
  /// The bunch-crossing ID to be used by the data packing methods
  uint32_t m_packingBxId;
  
  /// The event ID to be used by the data packing methods
  uint32_t m_packingEventId;

};

#endif /* GctFormatTranslateBase_h_ */
