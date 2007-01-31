
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
  GctBlockHeader();
  GctBlockHeader(const unsigned char * data);
  ~GctBlockHeader();
  
  std::vector<unsigned char> data() const { return d; }

  unsigned int id() const { return d[0]; }
  unsigned int nSamples() const { return d[1]&0xf; }
  unsigned int bcId() const { return ((d[1]&0xf0)>>4) + (d[2]<<4); }
  unsigned int eventId() const { return d[3]; }

  unsigned int blockLength() const;

  friend std::ostream& operator<<(std::ostream& os, const GctBlockHeader& h);

 private:
  
  std::vector<unsigned char> d;

};



#endif
