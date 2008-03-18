#ifndef GCTBLOCKUNPACKER_H
#define GCTBLOCKUNPACKER_H

#include "EventFilter/GctRawToDigi/src/GctBlockUnpackerBase.h"

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
  
  // get digis from block
  void convertBlock(const unsigned char * d, const GctBlockHeaderBase& hdr);


private:
 
  /// Function pointer typdef to a block unpack function.
  typedef void (GctBlockUnpacker::*PtrToUnpackFn)(const unsigned char *, const GctBlockHeaderBase&);
  /// Typedef for a block ID to unpack function map.
  typedef std::map<unsigned int, PtrToUnpackFn> BlockIdToUnpackFnMap;

  // PRIVATE METHODS
  // convert functions for each type of block
  /// unpack GCT EM Candidates and energy sums.
  void blockToGctEmCandsAndEnergySums(const unsigned char * d, const GctBlockHeaderBase& hdr);

  /// unpack GCT internal EM Candidates
  void blockToGctInternEmCand(const unsigned char * d, const GctBlockHeaderBase& hdr);

  /// unpack RCT EM Candidates
  void blockToRctEmCand(const unsigned char * d, const GctBlockHeaderBase& hdr);

  /// unpack Fibres
  void blockToFibres(const unsigned char * d, const GctBlockHeaderBase& hdr);
  
  /// unpack Fibres and RCT EM Candidates
  void blockToFibresAndToRctEmCand(const unsigned char * d, const GctBlockHeaderBase& hdr);
  
  /// Unpack GCT Jet Candidates and jet counts.
  void blockToGctJetCandsAndCounts(const unsigned char * d, const GctBlockHeaderBase& hdr);

  /// Do nothing
  void blockDoNothing(const unsigned char * d, const GctBlockHeaderBase& hdr) {}

};

#endif

