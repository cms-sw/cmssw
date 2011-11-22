#ifndef HcalDigi_HcalUpgradePackingScheme_h
#define HcalDigi_HcalUpgradePackingScheme_h

#include "EventFilter/HcalRawToDigi/interface/HcalUpgradeDataFramePacker.h"

class HcalUpgradePackingScheme
{
public:
  HcalUpgradePackingScheme();

  void pack(const HcalUpgradeDataFrame & frame, unsigned char * data) const;
  void unpack(const unsigned char * data, HcalUpgradeDataFrame & frame) const;

private:
  HcalUpgradeDataFramePacker hbPacker_;
  HcalUpgradeDataFramePacker hePacker_;
  HcalUpgradeDataFramePacker hfPacker_;
};

#endif

