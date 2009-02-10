#include "RecoLocalCalo/HcalRecAlgos/interface/HcalIgnoreCellsAlgo.h"

HcalIgnoreCellsAlgo::HcalIgnoreCellsAlgo()
{
  // default constructor
  statusMask_=0;
}


HcalIgnoreCellsAlgo::HcalIgnoreCellsAlgo(int mask)
{
  statusMask_=mask;
}

HcalIgnoreCellsAlgo::~HcalIgnoreCellsAlgo(){}

bool HcalIgnoreCellsAlgo::ignoreBadCells(DetId& id,  HcalChannelQuality* myqual)
{
  // return 'true' if cell status is marked as bad somehow in database; otherwise, return 'false'

  const HcalChannelStatus* mydigistatus=myqual->getValues(id);

  // Check for Cells that are off
  if ((statusMask_>>0) & 0x1)
    {
      if (mydigistatus->isBitSet(HcalChannelStatus::HcalCellOff)) return true;
    }
  // Check for Dead Cells
  if ((statusMask_>>5) & 0x1)
    {
      if (mydigistatus->isBitSet(HcalChannelStatus::HcalCellDead)) return true;
    }
  // Check for Hot Cells
  if ((statusMask_>>6) & 0x1)
    {
      if (mydigistatus->isBitSet(HcalChannelStatus::HcalCellHot)) return true;
    }
  // Check for Stability Errors
  if ((statusMask_>>7) & 0x1)
    {
      if (mydigistatus->isBitSet(HcalChannelStatus::HcalCellStabErr)) return true;
    }
  // Check for Timing Errors
  if ((statusMask_>>8) & 0x1)
    {
      if (mydigistatus->isBitSet(HcalChannelStatus::HcalCellTimErr)) return true;
    }
  return false;
} // bool HcalHitReconstructor::ignoreBadCells(...)

