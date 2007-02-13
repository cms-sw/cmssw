
#ifndef GCTDAQRECORD_H
#define GCTDAQRECORD_H

#include <vector>
#include <iostream>

#include "EventFilter/GctRawToDigi/src/GctBlock.h"


class GctDaqRecord {

 public:
  GctDaqRecord();
  GctDaqRecord(const unsigned char * data, const unsigned int size);
  ~GctDaqRecord();

  unsigned id() const { return header_[4] + (header_[5]<<8) + (header_[6]<<16); }
  unsigned l1Type() const { return header_[7]&0xf; }
  unsigned bcId() const { return header_[2]>>4 + (header_[3]<<4); }
  unsigned sourceId() const { return header_[1] + (header_[2]<<4)&0xf00; }

  const std::vector<GctBlock>& blocks() const { return blocks_; }

  friend std::ostream& operator<<(std::ostream& os, const GctDaqRecord& e);

 private:

  std::vector<unsigned char> header_;
  std::vector<unsigned char> footer_;

  std::vector<GctBlockHeader> blockHeaders_;
  std::vector<GctBlock> blocks_;
  
};


#endif
