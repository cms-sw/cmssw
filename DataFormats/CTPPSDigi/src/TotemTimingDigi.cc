/** \file
 * 
 *
 * \author Mirko Berretti
 * \author Nicola Minafra
 */

#include "DataFormats/CTPPSDigi/interface/TotemTimingDigi.h"

TotemTimingDigi::TotemTimingDigi(const uint8_t hwId, const uint64_t FPGATimeStamp, const uint16_t TimeStampA, const uint16_t TimeStampB, const uint16_t CellInfo, const std::vector<uint8_t>& Samples, const TotemTimingEventInfo& totemTimingEventInfo ) :
  hwId_(hwId), FPGATimeStamp_(FPGATimeStamp), TimeStampA_(TimeStampA), TimeStampB_(TimeStampB), CellInfo_(CellInfo), samples_(Samples), totemTimingEventInfo_(totemTimingEventInfo)
{}

TotemTimingDigi::TotemTimingDigi(const TotemTimingDigi& digi) :
  hwId_(digi.getHardwareId()), FPGATimeStamp_(digi.getFPGATimeStamp()), TimeStampA_(digi.getTimeStampA()), TimeStampB_(digi.getTimeStampB()), CellInfo_(digi.getCellInfo()), samples_(digi.getSamples()), totemTimingEventInfo_(digi.getEventInfo())
{}

TotemTimingDigi::TotemTimingDigi() :
  hwId_(0), FPGATimeStamp_(0), TimeStampA_(0), TimeStampB_(0), CellInfo_(0)
{}

// Comparison
bool
TotemTimingDigi::operator==(const TotemTimingDigi& digi) const
{
  if ( hwId_                    !=      digi.getHardwareId()
    || FPGATimeStamp_           !=      digi.getFPGATimeStamp()
    || TimeStampA_              !=      digi.getTimeStampA()
    || TimeStampB_              !=      digi.getTimeStampB()
    || CellInfo_                !=      digi.getCellInfo()
    || samples_                 !=      digi.getSamples()
  ) return false;
  else  
    return true; 
} 

