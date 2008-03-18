#ifndef GCTBLOCKHEADER_H
#define GCTBLOCKHEADER_H

#include "EventFilter/GctRawToDigi/src/GctBlockHeaderBase.h"

/// Class representing the first definition of a pipeline block header (GREN 2007 era - kept for backwards compat.) 
/*! blockId = 7:0,
 *  nSamples = 11:8 (if nSamples=0xf, use defNSamples_),
 *  bcId = 23:12,
 *  eventId = 31:24 */
class GctBlockHeader : public GctBlockHeaderBase
{
public:

  GctBlockHeader(const uint32_t data=0):GctBlockHeaderBase(data) {}
  GctBlockHeader(const unsigned char * data):GctBlockHeaderBase(data) {}
  GctBlockHeader(uint16_t id, uint16_t nsamples, uint16_t bcid, uint16_t evid);
  ~GctBlockHeader() {}
  
  /// the block ID
  unsigned int id() const { return d & 0xff; }

  /// number of time samples
  unsigned int nSamples() const { return (d>>8) & 0xf; }
  
  /// bunch crossing ID
  unsigned int bcId() const { return (d>>12) & 0xfff; }

  /// event ID
  unsigned int eventId() const { return (d>>24) & 0xff; }

protected:

  BlockLengthMap& blockLengthMap() { return blockLength_; }
  
  /// Pure virtual interface for accessing concrete-subclass static blockname map.
  BlockNameMap& blockNameMap() { return blockName_; }
  

private:

  /// Map to translate block number to fundamental size of a block (i.e. for 1 time-sample).
  static BlockLengthMap blockLength_;
  
  /// Map to hold a description for each block number.
  static BlockNameMap blockName_;
};

#endif
