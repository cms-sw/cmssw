
#ifndef GCTBLOCKHEADER_H
#define GCTBLOCKHEADER_H

#include <vector>
#include <map>
#include <ostream>
#include <string>

// data block header
// blockId = 7:0
// nSamples = 11:8 (if nSamples=0xf, use defNSamples_)
// bcId = 23:12
// eventId = 31:24



class GctBlockHeader {
 public:
  GctBlockHeader(const uint32_t data=0);
  GctBlockHeader(const unsigned char * data);
  GctBlockHeader(uint16_t id, uint16_t nsamples, uint16_t bcid, uint16_t evid);
  ~GctBlockHeader();
  
  /// this is a valid block header
  bool valid() const { return ( blockLength_.find(this->id()) != blockLength_.end() ); }

  /// the raw header data
  uint32_t data() const { return d; }

  /// the block ID
  unsigned int id() const { return d & 0xff; }

  /// number of time samples
  unsigned int nSamples() const { return (d>>8) & 0xf; }
  
  /// bunch crossing ID
  unsigned int bcId() const { return (d>>12) & 0xfff; }

  /// event ID
  unsigned int eventId() const { return (d>>24) & 0xff; }

  /// fundamental block length (for 1 time sample)
  unsigned int length() const { return blockLength_[this->id()] ; }

  /// block name
  std::string name() const { return blockName_[this->id()]; }

  // Static method for looking up block lengths for a given block ID
  static unsigned int lookupBlockLength(unsigned int blockId);

 private:
  
  uint32_t d;

  static std::map<unsigned int, unsigned int> blockLength_;  // fundamental size of a block (ie for 1 readout sample)
  static std::map<unsigned int, std::string> blockName_;  // block name!



};

std::ostream& operator<<(std::ostream& os, const GctBlockHeader& h);



#endif
