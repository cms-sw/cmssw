#ifndef GCTBLOCKHEADERV2_H_
#define GCTBLOCKHEADERV2_H_

#include "EventFilter/GctRawToDigi/src/GctBlockHeaderBase.h"


/*!
* \class GctBlockHeaderV2
* \brief Version 2 of the pipeline block header, as specified by Greg Iles on the 11-1-2008:
* 
*  Block Header Version 2... complies with Pipeline Formats v20 and
*  is up to date with the hardware as of 19th March 2008.
* 
* \author Robert Frazier
* $Revision: $
* $Date: $
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
