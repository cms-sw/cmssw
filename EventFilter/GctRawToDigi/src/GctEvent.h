
#ifndef GCTEVENT_H
#define GCTEVENT_H

#include <vector>
#include <ostream>

#include "EventFilter/GctRawToDigi/src/GctBlock.h"


class GctEvent {

 public:
  GctEvent(const unsigned char * data, const unsigned int size);
  ~GctEvent();

  unsigned id() const { return  header_[4] + (header_[5]<<8) + (header_[6]<<16); }
  unsigned l1Type() const { return  header_[7]<<4; }
  unsigned bcId() const { return header_[2]>>4 + (header_[3]<<4); }
  unsigned sourceId() const { return header_[1] + (header_[2]<<4)&0xf00; }

  const std::vector<GctBlock>& blocks() const { return blocks_; }

  friend std::ostream& operator<<(std::ostream& os, const GctEvent& e);

 private:

  unsigned char header_[8];
  unsigned char footer_[8];

  std::vector<GctBlock> blocks_;

};


#endif
