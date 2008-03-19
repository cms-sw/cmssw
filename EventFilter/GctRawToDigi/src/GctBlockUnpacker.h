#ifndef GCTBLOCKUNPACKER_H
#define GCTBLOCKUNPACKER_H

#include "EventFilter/GctRawToDigi/src/GctBlockUnpackerBase.h"

/*!
* \class GctBlockUnpacker
* \brief Original (now deprecated) concrete unpacker for GREN 2007 era data.
* 
* 
* \author Robert Frazier
* $Revision: $
* $Date: $
*/ 


// *******************************************************************
// ***  THE UNPACK PROCESS MUST NEVER THROW ANY KIND OF EXCEPTION! *** 
// *******************************************************************

class GctBlockUnpacker : public GctBlockUnpackerBase
{
public:

  /// Constructor.
  /*! \param hltMode - set true to unpack only BX zero and GCT output data (i.e. to run as quickly as possible).*/
  GctBlockUnpacker(bool hltMode = false);
  
  ~GctBlockUnpacker(); ///< Destructor.
  
  /// Get digis from the block.
  void convertBlock(const unsigned char * d, const GctBlockHeaderBase& hdr);


private:

  // PRIVATE TYPEDEFS
 
  /// Function pointer typdef to a block unpack function.
  typedef void (GctBlockUnpacker::*PtrToUnpackFn)(const unsigned char *, const GctBlockHeaderBase&);
  /// Typedef for a block ID to unpack function map.
  typedef std::map<unsigned int, PtrToUnpackFn> BlockIdToUnpackFnMap;


  // PRIVATE MEMBER DATA
  
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
  
  /// Unpack GCT Jet Counts
  void blockToGctJetCounts(const unsigned char * d, const GctBlockHeaderBase& hdr);
  
};

#endif

