#ifndef GCTBLOCKUNPACKERV2_H
#define GCTBLOCKUNPACKERV2_H

#include "EventFilter/GctRawToDigi/src/GctBlockUnpackerBase.h"

/*!
* \class GctBlockUnpackerV2
* \brief Second version of the block unpacker - unpacks current hardware output.
* 
*  Block Unpacker Version 2... complies with Pipeline Formats v20 and
*  is up to date with the hardware as of 19th March 2008.
* 
* \author Robert Frazier
* $Revision: 1.3 $
* $Date: 2008/03/19 16:14:57 $
*/ 

// *******************************************************************
// ***  THE UNPACK PROCESS MUST NEVER THROW ANY KIND OF EXCEPTION! *** 
// *******************************************************************

class GctBlockUnpackerV2 : public GctBlockUnpackerBase
{
public:

  /// Constructor.
  /*! \param hltMode - set true to unpack only BX zero and GCT output data (i.e. to run as quickly as possible).*/
  GctBlockUnpackerV2(bool hltMode = false);
  
  ~GctBlockUnpackerV2(); ///< Destructor.
  
  /// Get digis from the block - will return true if it succeeds, false otherwise.
  bool convertBlock(const unsigned char * d, const GctBlockHeaderBase& hdr);


protected:
  
  // PROTECTED METHODS
  
  RctCrateMap& rctCrateMap() { return rctCrate_; }  ///< get the RCT crate map.
  const RctCrateMap& rctCrateMap() const { return rctCrate_; }  ///< get the RCT crate map.
  
  BlockIdToEmCandIsoBoundMap& internEmIsoBounds() { return internEmIsoBounds_; }  ///< get the intern EM cand isolated boundary map.
  const BlockIdToEmCandIsoBoundMap& internEmIsoBounds() const { return internEmIsoBounds_; }  ///< get the intern EM cand isolated boundary map.

private:
  // PRIVATE TYPEDEFS
 
  /// Function pointer typdef to a block unpack function.
  typedef void (GctBlockUnpackerV2::*PtrToUnpackFn)(const unsigned char *, const GctBlockHeaderBase&);
  /// Typedef for a block ID to unpack function map.
  typedef std::map<unsigned int, PtrToUnpackFn> BlockIdToUnpackFnMap;


  // PRIVATE MEMBER DATA
  
  /// Map to relate capture block ID to the RCT crate the data originated from.
  static RctCrateMap rctCrate_;
  
  /*! A map of Block IDs to IsoBoundaryPairs for storing the location of the isolated
   *  Internal EM cands in the pipeline, as this differs with Block ID. */ 
  static BlockIdToEmCandIsoBoundMap internEmIsoBounds_;

  /// Block ID to unpack function map.
  static BlockIdToUnpackFnMap blockUnpackFn_;
  

  // PRIVATE METHODS
  
  // Convert functions for each type of block
  /// unpack GCT EM Candidates and energy sums.
  void blockToGctEmCandsAndEnergySums(const unsigned char * d, const GctBlockHeaderBase& hdr);
  
  /// Unpack GCT Jet Candidates and jet counts.
  void blockToGctJetCandsAndCounts(const unsigned char * d, const GctBlockHeaderBase& hdr);
};

#endif
