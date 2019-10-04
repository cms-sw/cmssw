#ifndef DATAFORMATS_HCALDIGI_HCALLASERDIGI_H
#define DATAFORMATS_HCALDIGI_HCALLASERDIGI_H 1

#include <vector>
#include <cstdint>

class HcalLaserDigi {
public:
  HcalLaserDigi();
  void setQADC(const std::vector<uint16_t>& values);
  uint16_t qadc(int i) const { return ((i >= 0 && i < 32) ? (qadcraw_[i]) : (0)); }
  void addTDCHit(int channel, int hittime);
  size_t tdcHits() const { return tdcraw_.size(); }
  int hitChannel(size_t ihit) const;
  int hitRaw(size_t ihit) const;
  double hitNS(size_t ihit) const;
  void setLaserControl(int att1, int att2, int select);
  int attenuator1() const { return attenuator1_; }
  int attenuator2() const { return attenuator2_; }
  int selector() const { return selector_; }

private:
  uint16_t qadcraw_[32];
  std::vector<uint32_t> tdcraw_;
  int32_t attenuator1_, attenuator2_;
  int32_t selector_;
};

#endif  // DATAFORMATS_HCALDIGI_HCALLASERDIGI_H
