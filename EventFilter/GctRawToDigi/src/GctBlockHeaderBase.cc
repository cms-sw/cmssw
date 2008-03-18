#include "EventFilter/GctRawToDigi/src/GctBlockHeaderBase.h"

GctBlockHeaderBase::GctBlockHeaderBase(const unsigned char * data)
{ 
  d = data[0] + (data[1]<<8) + (data[2]<<16) + (data[3]<<24);
}

unsigned int GctBlockHeaderBase::length() const
{
  if(!valid()) { return 0; }
  return blockLength_[this->id()];
}

std::string GctBlockHeaderBase::name() const
{
  if(!valid()) { return "Unknown/invalid block header"; }
  return blockName_[this->id()];
}

std::ostream& operator<<(std::ostream& os, const GctBlockHeaderBase& h)
{
  os << "GCT Raw Data Block : " << h.name() << " : ID " << std::hex << h.id()
     << " : Length : " << h.length() << " : Samples " << h.nSamples() << " : BX "
     << h.bcId() << " : Event " << h.eventId() << std::dec;
  return os;
}
