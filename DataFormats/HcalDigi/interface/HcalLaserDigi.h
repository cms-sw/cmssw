#ifndef DATAFORMATS_HCALDIGI_HCALLASERDIGI_H
#define DATAFORMATS_HCALDIGI_HCALLASERDIGI_H 1

#include <boost/cstdint.hpp>
#include <vector>

class HcalLaserDigi {
public:
  HcalLaserDigi();
  void setQADC(const std::vector<uint16_t>& values);
  uint16_t qadc(int i) const { return ((i>=0 && i<32)?(qadcraw_[i]):(0)); }
  void addTDCHit(int channel, int hittime);
  size_t tdcHits() const { return tdcraw_.size(); }
  int hitChannel(size_t ihit) const;
  int hitRaw(size_t ihit) const;
  double hitNS(size_t ihit) const;
private:	      
  uint16_t qadcraw_[32];
  std::vector<uint32_t> tdcraw_;
};

#endif // DATAFORMATS_HCALDIGI_HCALLASERDIGI_H
