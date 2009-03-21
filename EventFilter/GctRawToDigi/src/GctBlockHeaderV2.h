#ifndef GCTBLOCKHEADERV2_H_
#define GCTBLOCKHEADERV2_H_

#include "EventFilter/GctRawToDigi/src/GctBlockHeaderBase.h"


/*!
* \class GctBlockHeaderV2
* \brief Version 2 of the pipeline block header, as specified by Greg Iles on the 11-1-2008:
* 
*  Block Header Version 2... For use with Pipeline Formats v20 and above.
* 
* \author Robert Frazier
* $Revision: 1.9 $
* $Date: 2009/03/16 16:59:39 $
*/ 



//  Bit mapping of the header:
//  --------------------------
//  11:0   => block_id  Unique pipeline identifier.
//   - 3:0    =>> pipe_id There can be up to 16 different pipelines per FPGA.
//   - 6:4    =>> reserved  Do not use yet. Set to zero.
//   - 11:7   =>> fpga geograpical add  The VME geographical address of the FPGA.
//  15:12  => event_id  Determined locally.  Not reset by Resync.
//  19:16  => number_of_time_samples  If time samples 15 or more then value = 15.
//  31:20  => event_bcid  The bunch crossing the data was recorded.


class GctBlockHeaderV2 : public GctBlockHeaderBase
{
public:

  /// Enum for use with initBlockLengthMap static method.
  enum BlockLengthMapVersion{ BLOCK_LENGTHS_FOR_UNPACKER_V2, BLOCK_LENGTHS_FOR_UNPACKER_V3 }; 

  GctBlockHeaderV2(const uint32_t data=0):GctBlockHeaderBase(data) {}
  GctBlockHeaderV2(const unsigned char * data):GctBlockHeaderBase(data) {}
  GctBlockHeaderV2(uint16_t id, uint16_t nsamples, uint16_t bcid, uint16_t evid);
  ~GctBlockHeaderV2() {}
  
  /// the block ID
  unsigned int id() const { return d & 0xfff; }

  /// number of time samples
  unsigned int nSamples() const { return (d>>16) & 0xf; }
  
  /// bunch crossing ID
  unsigned int bcId() const { return (d>>20) & 0xfff; }

  /// event ID
  unsigned int eventId() const { return (d>>12) & 0xf; }

  /// Hacktastic static method to init the blockLengthMap with the correct lengths for the raw format in use.
  static void initBlockLengthMap(const BlockLengthMapVersion lengthMapVersion); 

protected:

  BlockLengthMap& blockLengthMap() { return blockLengthV2_; }
  const BlockLengthMap& blockLengthMap() const { return blockLengthV2_; }
  
  /// Pure virtual interface for accessing concrete-subclass static blockname map.
  BlockNameMap& blockNameMap() { return blockNameV2_; }
  const BlockNameMap& blockNameMap() const { return blockNameV2_; }
  

private:

  /// Map to translate block number to fundamental size of a block (i.e. for 1 time-sample).
  static BlockLengthMap blockLengthV2_;
  
  /// Map to hold a description for each block number.
  static BlockNameMap blockNameV2_;
};

#endif /*GCTBLOCKHEADERV2_H_*/
