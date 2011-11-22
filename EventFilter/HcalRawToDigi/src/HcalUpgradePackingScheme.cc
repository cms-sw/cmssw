#include "EventFilter/HcalRawToDigi/interface/HcalUpgradePackingScheme.h"
#include "DataFormats/HcalDigi/interface/HcalUpgradeDataFrame.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "FWCore/Utilities/interface/Exception.h"

HcalUpgradePackingScheme::HcalUpgradePackingScheme()
{
  unsigned tdcRisingPosHB[] = {66, 71, 77, 82};
  std::vector<unsigned> tdcRisingPosVHB(tdcRisingPosHB, tdcRisingPosHB+4);
  std::vector<unsigned> emptyV;
  hbPacker_ = HcalUpgradeDataFramePacker(8, tdcRisingPosVHB, emptyV, 64, 70, 76);

  unsigned tdcRisingPosHE[] = {50, 55, 60, 66, 71, 76};
  std::vector<unsigned> tdcRisingPosVHE(tdcRisingPosHE, tdcRisingPosHE+6);
  hePacker_ = HcalUpgradeDataFramePacker(6, tdcRisingPosVHE, emptyV, 48, 65, 81);

  // assume only 12 extra bits
  unsigned tdcRisingPosHF[] = {34, 39, 66, 71};
  unsigned tdcFallingPosHF[] = {44, 49, 77, 82};
  std::vector<unsigned> tdcRisingPosVHF(tdcRisingPosHF, tdcRisingPosHF+4);
  std::vector<unsigned> tdcFallingPosVHF(tdcFallingPosHF, tdcFallingPosHF+4);
  hfPacker_ = HcalUpgradeDataFramePacker(4, tdcRisingPosVHF, tdcFallingPosVHF, 32, 76, 87);
}

void HcalUpgradePackingScheme::pack(const HcalUpgradeDataFrame & frame, unsigned char * data) const
{
  switch(frame.id().subdetId())
    {
    case HcalBarrel:
      hbPacker_.pack(frame, data);
      break;
    case HcalEndcap:
      hePacker_.pack(frame, data);
      break;
    case HcalOuter:
      hbPacker_.pack(frame, data);
      break;
    case HcalForward:
      hfPacker_.pack(frame, data);
      break;
    default:
      throw cms::Exception("HcalUpgradePackingScheme") << "Bad Hcal subdetector " << frame.id().subdetId();
      break;
    }
}

void HcalUpgradePackingScheme::unpack(const unsigned char * data, HcalUpgradeDataFrame & frame) const
{
  switch(frame.id().subdetId())
    {
    case HcalBarrel:
      hbPacker_.unpack(data, frame);
      break;
    case HcalEndcap:
      hePacker_.unpack(data, frame);
      break;
    case HcalOuter:
      hbPacker_.unpack(data, frame);
      break;
    case HcalForward:
      hfPacker_.unpack(data, frame);
      break;
    default:
      throw cms::Exception("HcalUpgradePackingScheme") << "Bad Hcal subdetector " << frame.id().subdetId();
      break;
    }

}

