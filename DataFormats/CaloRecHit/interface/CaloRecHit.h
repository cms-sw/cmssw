#ifndef DATAFORMATS_CALORECHIT_CALORECHIT_H
#define DATAFORMATS_CALORECHIT_CALORECHIT_H 1

#include "DataFormats/DetId/interface/DetId.h"
#include <ostream>

#ifdef __CUDA_ARCH__
  __constant__
#else
  constexpr
#endif
  uint32_t calo_rechit_masks[] = {
    0x00000000u,0x00000001u,0x00000003u,0x00000007u,0x0000000fu,0x0000001fu,
    0x0000003fu,0x0000007fu,0x000000ffu,0x000001ffu,0x000003ffu,0x000007ffu,
    0x00000fffu,0x00001fffu,0x00003fffu,0x00007fffu,0x0000ffffu,0x0001ffffu,
    0x0003ffffu,0x0007ffffu,0x000fffffu,0x001fffffu,0x003fffffu,0x007fffffu,
    0x00ffffffu,0x01ffffffu,0x03ffffffu,0x07ffffffu,0x0fffffffu,0x1fffffffu,
    0x3fffffffu,0x7fffffffu,0xffffffffu};

/** \class CaloRecHit
 * 
 *\author J. Mans - Minnesota
 */
class CaloRecHit {
public:
  constexpr CaloRecHit() : energy_(0), time_(0), flags_(0), aux_(0) {}
  constexpr explicit CaloRecHit(const DetId& id, float energy, float time, 
                                uint32_t flags = 0, uint32_t aux=0)
	: id_(id),energy_(energy), time_(time), flags_(flags), aux_(aux) {}

  constexpr float energy() const { return energy_; }
  constexpr void setEnergy(float energy) { energy_=energy; }
  constexpr float time() const { return time_; }
  constexpr void setTime(float time) { time_=time; }
  constexpr const DetId& detid() const { return id_; }
  constexpr uint32_t flags() const { return flags_; }
  constexpr void setFlags(uint32_t flags) { flags_=flags; }
  constexpr void setFlagField(uint32_t value, int base, int width=1) {
    value&=calo_rechit_masks[std::max(std::min(width,32),0)];
    value<<=std::max(std::min(base,31),0);
    // clear out the relevant bits
    uint32_t clear=calo_rechit_masks[std::max(std::min(width,32),0)];
    clear=clear<<std::max(std::min(base,31),0);
    clear^=0xFFFFFFFFu;
    flags_&=clear;
    flags_|=value;
  }
  constexpr uint32_t flagField(int base, int width=1) const {
    return (flags_>>std::max(std::min(base,31),0))&calo_rechit_masks[std::max(std::min(width,32),0)];
  }
  constexpr void setAux(uint32_t value) { aux_=value; }
  constexpr uint32_t aux() const { return aux_; }
private:
  DetId id_;
  float energy_;
  float time_;
  uint32_t flags_;
  uint32_t aux_;
};

std::ostream& operator<<(std::ostream& s, const CaloRecHit& hit);
  
#endif
