
#ifndef GCTBLOCKHEADER_H
#define GCTBLOCKHEADER_H

#include <vector>
#include <ostream>

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

 private:
  
  uint32_t d;

};

std::ostream& operator<<(std::ostream& os, const GctBlockHeader& h);



#endif
