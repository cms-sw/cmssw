
#ifndef GCTBLOCKHEADER_H
#define GCTBLOCKHEADER_H

#include <ostream>


// data block header
// blockId = 7:0
// nSamples = 11:8 (if nSamples=0xf, use defNSamples_)
// bcId = 23:12
// eventId = 31:24



class GctBlockHeader {
 public:
  GctBlockHeader();
  GctBlockHeader(const unsigned char * data);
  ~GctBlockHeader();
  
  const unsigned char * data() const { return d; }

  unsigned char id() const { return d[0]; }
  unsigned char length() const { return d[1]&0xf; }
  unsigned int bcId() const { return ((d[1]&0xf0)>>4) + (d[2]<<4); }
  unsigned char eventId() const { return d[3]; }

  friend std::ostream& operator<<(std::ostream& os, const GctBlockHeader& h);

 private:
  
  const unsigned char * d;

};



#endif
