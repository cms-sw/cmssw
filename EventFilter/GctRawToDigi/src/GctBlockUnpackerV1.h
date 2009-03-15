#ifndef GCTBLOCKUNPACKERV1_H
#define GCTBLOCKUNPACKERV1_H

#include "EventFilter/GctRawToDigi/src/GctBlockUnpackerBase.h"

/*!
* \class GctBlockUnpackerV1
* \brief Original (now deprecated) concrete unpacker for GREN 2007 era data.
* 
* 
* \author Robert Frazier
* $Revision: 1.23 $
* $Date: 2009/03/15 16:05:50 $
*/ 


// *******************************************************************
// ***  THE UNPACK PROCESS MUST NEVER THROW ANY KIND OF EXCEPTION! *** 
// *******************************************************************

class GctBlockUnpackerV1 : public GctBlockUnpackerBase
{
public:

  /// Constructor.
  /*! \param hltMode - set true to unpack only BX zero and GCT output data (i.e. to run as quickly as possible).*/
  GctBlockUnpackerV1(bool hltMode = false);
  
  ~GctBlockUnpackerV1(); ///< Destructor.
  
  /// Get digis from the block - will return true if it succeeds, false otherwise.
  bool convertBlock(const unsigned char * d, const GctBlockHeaderBase& hdr);


protected:
  
  // PROTECTED METHODS
  
  RctCrateMap& rctCrateMap() { return rctCrate_; }  ///< get the RCT crate map.
  const RctCrateMap& rctCrateMap() const { return rctCrate_; }  ///< get the RCT crate map.

  RctCrateMap& rctJetCrateMap() { return rctJetCrate_; }  ///< get the RCT jet crate map.
  const RctCrateMap& rctJetCrateMap() const { return rctJetCrate_; }  ///< get the RCT jet crate map.

  BlockIdToEmCandIsoBoundMap& internEmIsoBounds() { return internEmIsoBounds_; }  ///< get the intern EM cand isolated boundary map.
  const BlockIdToEmCandIsoBoundMap& internEmIsoBounds() const { return internEmIsoBounds_; }  ///< get the intern EM cand isolated boundary map.

private:

  // PRIVATE TYPEDEFS
 
  /// Function pointer typdef to a block unpack function.
  typedef void (GctBlockUnpacker::*PtrToUnpackFn)(const unsigned char *, const GctBlockHeaderBase&);
  /// Typedef for a block ID to unpack function map.
  typedef std::map<unsigned int, PtrToUnpackFn> BlockIdToUnpackFnMap;


  // PRIVATE MEMBER DATA

  /// Map to relate capture block ID to the RCT crate the data originated from.
  static RctCrateMap rctCrate_;
  static RctJetCrateMap rctJetCrate_;

  /*! A map of Block IDs to IsoBoundaryPairs for storing the location of the isolated
   *  Internal EM cands in the pipeline, as this differs with Block ID. */ 
  static BlockIdToEmCandIsoBoundMap internEmIsoBounds_;

  /// Block ID to unpack function map.
  static BlockIdToUnpackFnMap blockUnpackFn_;
  

  // PRIVATE METHODS
  
  // Convert functions for each type of block
  /// unpack GCT EM Candidates
  void blockToGctEmCand(const unsigned char * d, const GctBlockHeaderBase& hdr);
  
  /// Unpack GCT Energy Sums (Et, Ht, and Missing Et)
  void blockToGctEnergySums(const unsigned char * d, const GctBlockHeaderBase& hdr);
  
  /// Unpack GCT Jet Candidates.
  void blockToGctJetCand(const unsigned char * d, const GctBlockHeaderBase& hdr);
  
};

#endif

