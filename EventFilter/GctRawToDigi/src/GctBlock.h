
#ifndef GCTBLOCK_H
#define GCTBLOCK_H

#include <vector>
#include <ostream>

#include "EventFilter/GctRawToDigi/src/GctBlockHeader.h"

// GCT data block

class GctBlock {
 public:
  GctBlock(const unsigned char * data);
  ~GctBlock();
  
  unsigned char id() const { return head.id(); }
  unsigned char nSamples() const { return head.nSamples(); }
  unsigned int bcId() const { return head.bcId(); }
  unsigned char eventId() const { return head.eventId(); }
  unsigned char length() const { return head.blockLength(); }

  std::vector<unsigned char> data() const { return d; }

  friend std::ostream& operator<<(std::ostream& os, const GctBlock& b);

 private:
  
  GctBlockHeader head;
  std::vector<unsigned char> d;

};

#endif
