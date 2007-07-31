
#ifndef GCTBLOCKHEADER_H
#define GCTBLOCKHEADER_H

#include <vector>
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
  
  uint32_t data() const { return d; }

  unsigned int id() const { return d & 0xff; }
  unsigned int nSamples() const { return (d>>8) & 0xf; }
  unsigned int bcId() const { return (d>>12) & 0xfff; }
  unsigned int eventId() const { return (d>>24) & 0xff; }

  //  unsigned int length() const { return blockLength_.find(id()); }

 private:
  
  uint32_t d;

  static std::map<unsigned, unsigned> blockLength_;  // fundamental size of a block (ie for 1 readout sample)
  static std::map<unsigned, std::string> blockName_;  // block name!



};

std::ostream& operator<<(std::ostream& os, const GctBlockHeader& h);



#endif
