#ifndef HcalDigi_HcalUpgradeDataFramePacker_h
#define HcalDigi_HcalUpgradeDataFramePacker_h

class HcalUpgradeDataFrame;
#include<vector>

class HcalUpgradeDataFramePacker
{
public:
  enum {NBYTES=11};
  HcalUpgradeDataFramePacker() {}
  HcalUpgradeDataFramePacker(unsigned nadc,
    const std::vector<unsigned> & tdcRisingPos,
    const std::vector<unsigned> & tdcFallingPos,
    unsigned capIdPos, unsigned errbit0Pos, unsigned errbit1Pos);

  void pack(const HcalUpgradeDataFrame & frame, unsigned char * data) const;
  void unpack(const unsigned char * data, HcalUpgradeDataFrame & frame) const; 

private:
  unsigned nadc_;
  std::vector<unsigned> tdcRisingPos_;
  unsigned ntdcRising_;
  std::vector<unsigned> tdcFallingPos_;
  unsigned ntdcFalling_;
  unsigned capIdPos_;
  unsigned errbit0Pos_;
  unsigned errbit1Pos_;
};

#endif

