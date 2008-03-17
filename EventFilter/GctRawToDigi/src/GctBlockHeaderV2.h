#ifndef GCTBLOCKHEADERV2_H_
#define GCTBLOCKHEADERV2_H_

#include <vector>
#include <map>
#include <ostream>
#include <string>

// Version 2 of the pipeline block header, as specified by Greg Iles on the 11-1-2008:
// -----------------------------------------------------------------------------------
//
//  11:0   => block_id  Unique pipeline identifier
//   - 3:0    =>> pipe_id There can be up to 16 different pipelines per FPGA
//   - 6:4    =>> reserved  Do not use yet. Set to zero
//   - 11:7   =>> fpga geograpical add  The VME geographical address of the FPGA
//  15:12  => event_id  Determined locally.  Not reset by Resync
//  19:16  => number_of_time_samples  If time samples 15 or more then value = 15.
//  31:20  => event_bcid  The bunch crossing the data was recorded

class GctBlockHeaderV2 {
 public:
  GctBlockHeaderV2(const uint32_t data=0);
  GctBlockHeaderV2(const unsigned char * data);
  GctBlockHeaderV2(uint16_t id, uint16_t nsamples, uint16_t bcid, uint16_t evid);
  ~GctBlockHeaderV2();
  
  /// this is a valid block header
  bool valid() const { return ( blockLength_.find(this->id()) != blockLength_.end() ); }

  /// the raw header data
  uint32_t data() const { return d; }

  /// the block ID
  unsigned int id() const { return d & 0xfff; }

  /// number of time samples
  unsigned int nSamples() const { return (d>>16) & 0xf; }
  
  /// bunch crossing ID
  unsigned int bcId() const { return (d>>20) & 0xfff; }

  /// event ID
  unsigned int eventId() const { return (d>>12) & 0xf; }

  /// fundamental block length (for 1 time sample)
  unsigned int length() const;

  /// block name
  std::string name() const;

 private:
  
  uint32_t d;

  static std::map<unsigned int, unsigned int> blockLength_;  // fundamental size of a block (ie for 1 readout sample)
  static std::map<unsigned int, std::string> blockName_;  // block name!



};

std::ostream& operator<<(std::ostream& os, const GctBlockHeaderV2& h);











#endif /*GCTBLOCKHEADERV2_H_*/
