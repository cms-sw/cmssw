#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"


CaloRecHit::CaloRecHit() : energy_(0), time_(0), flags_(0), aux_(0) {
}

CaloRecHit::CaloRecHit(const DetId& id, float energy, float time, uint32_t flags, uint32_t aux) : 
	id_(id),energy_(energy), time_(time), flags_(flags), aux_(aux) {
}


static const uint32_t masks[] = {
  0x00000000u,0x00000001u,0x00000003u,0x00000007u,0x0000000fu,0x0000001fu,
  0x0000003fu,0x0000007fu,0x000000ffu,0x000001ffu,0x000003ffu,0x000007ffu,
  0x00000fffu,0x00001fffu,0x00003fffu,0x00007fffu,0x0000ffffu,0x0001ffffu,
  0x0003ffffu,0x0007ffffu,0x000fffffu,0x001fffffu,0x003fffffu,0x007fffffu,
  0x00ffffffu,0x01ffffffu,0x03ffffffu,0x07ffffffu,0x0fffffffu,0x1fffffffu,
  0x3fffffffu,0x7fffffffu,0xffffffffu};

void CaloRecHit::setFlagField(uint32_t value, int base, int width) {
  value&=masks[std::max(std::min(width,32),0)];
  value<<=std::max(std::min(base,31),0);
  // clear out the relevant bits
  uint32_t clear=masks[std::max(std::min(width,32),0)];
  clear=clear<<std::max(std::min(base,31),0);
  clear^=0xFFFFFFFFu;
  flags_&=clear;
  flags_|=value;
}

uint32_t CaloRecHit::flagField(int base, int width) const {
  return (flags_>>std::max(std::min(base,31),0))&masks[std::max(std::min(width,32),0)];
}


std::ostream& operator<<(std::ostream& s, const CaloRecHit& hit) {
  s << hit.detid().rawId() << ", " << hit.energy() << " GeV, " << hit.time() << " ns ";
  s << " flags=0x" << std::hex << hit.flags() << std::dec << " ";
  s << " aux=0x" << std::hex << hit.aux() << std::dec << " ";
  return s;
}

