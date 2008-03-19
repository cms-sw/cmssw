#ifndef GCTBLOCKUNPACKERV2_H
#define GCTBLOCKUNPACKERV2_H

#include "EventFilter/GctRawToDigi/src/GctBlockUnpackerBase.h"

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
  
  /// Get digis from the block.
  void convertBlock(const unsigned char * d, const GctBlockHeaderBase& hdr);

private:
  // PRIVATE TYPEDEFS
 
  /// Function pointer typdef to a block unpack function.
  typedef void (GctBlockUnpackerV2::*PtrToUnpackFn)(const unsigned char *, const GctBlockHeaderBase&);
  /// Typedef for a block ID to unpack function map.
  typedef std::map<unsigned int, PtrToUnpackFn> BlockIdToUnpackFnMap;


  // PRIVATE MEMBER DATA
  
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
