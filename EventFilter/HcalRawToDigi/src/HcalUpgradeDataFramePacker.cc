#include "EventFilter/HcalRawToDigi/interface/HcalUpgradeDataFramePacker.h"
#include "DataFormats/HcalDigi/interface/HcalUpgradeDataFrame.h"
#include <bitset>

HcalUpgradeDataFramePacker::HcalUpgradeDataFramePacker(unsigned nadc, 
  const std::vector<unsigned> & tdcRisingPos,
  const std::vector<unsigned> & tdcFallingPos,
  unsigned capIdPos, unsigned errbit0Pos, unsigned errbit1Pos)
: nadc_(nadc),
  tdcRisingPos_(tdcRisingPos),
  ntdcRising_(tdcRisingPos.size()),
  tdcFallingPos_(tdcFallingPos),
  ntdcFalling_(tdcFallingPos.size()),
  capIdPos_(capIdPos),
  errbit0Pos_(errbit0Pos),
  errbit1Pos_(errbit1Pos)
{
} 

void HcalUpgradeDataFramePacker::pack(const HcalUpgradeDataFrame & frame, unsigned char * data) const
{
  std::bitset<88> bits;
  std::bitset<88> mask8(0xFF);
  unsigned i = 0;
  // adcs first
  for(i = 0; i < nadc_; ++i)
  {
    std::bitset<88> tmpbits(frame.adc(i) & 0xFF);
    bits |= tmpbits << 8*i;
  }
  // capID
  std::bitset<88> tmpbits(frame.capId() & 0x3);
  bits |= tmpbits << capIdPos_;
  // tdc
  for(i = 0; i < ntdcRising_; ++i)
  {
    std::bitset<88> tmpbits(frame.tdc(i) & 0x1F);
    bits |= tmpbits << tdcRisingPos_[i];
  }
  for(i = 0; i < ntdcFalling_; ++i)
  {
    std::bitset<88> tmpbits((frame.tdc(i)>>8)&0x1F);
    bits |= tmpbits << tdcFallingPos_[i];
  }
  // copy into result
  for(i = 0; i < NBYTES; ++i) {
    data[i] = ((bits >> 8*i) & mask8).to_ulong();
  }
 
}

void HcalUpgradeDataFramePacker::unpack(const unsigned char * data, HcalUpgradeDataFrame & frame) const
{
  std::bitset<88> bits;
  std::bitset<88> mask2(0x3);
  std::bitset<88> mask5(0x1F);
  unsigned i = 0;
  for(i = 0; i < NBYTES; ++i) {
    std::bitset<88> tmpbits(data[i]);
    bits |= (tmpbits << 8*i);
  }
  frame.setSize(nadc_);
  unsigned capId = ((bits >> capIdPos_) & mask2).to_ulong();
  capId = ((bits >> capIdPos_) & mask2).to_ulong();

  frame.setStartingCapId(capId);
  for(i = 0; i < nadc_; ++i)
  {
    unsigned adc = data[i] & 0xFF;
    unsigned tdc = 0;
    if(i < ntdcRising_) 
    {
      tdc |= ((bits >> tdcRisingPos_[i]) & mask5).to_ulong();
    }
    if(i < ntdcFalling_)
    {
      tdc |= (((bits >> tdcFallingPos_[i]) & mask5) << 8).to_ulong();
    }
    frame.setSample(i, adc, tdc, true); 
  }
}

