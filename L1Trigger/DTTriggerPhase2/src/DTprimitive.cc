#include "L1Trigger/DTTriggerPhase2/interface/DTprimitive.h"
#include "L1Trigger/DTTriggerPhase2/interface/constants.h"
#include <iostream>
#include <iomanip>

using namespace cmsdt;
//------------------------------------------------------------------
//--- Constructors and destructor
//------------------------------------------------------------------
DTPrimitive::DTPrimitive() {
  cameraId_ = -1;
  superLayerId_ = -1;
  layerId_ = -1;
  channelId_ = -1;
  tdcTimeStamp_ = -1;
  orbit_ = -1;
  timeCorrection_ = 0;
  laterality_ = NONE;

  for (int i = 0; i < PAYLOAD_ENTRIES; i++)
    setPayload(0.0, i);
}

DTPrimitive::DTPrimitive(DTPrimitivePtr& ptr) {
  setTimeCorrection(ptr->timeCorrection());
  setTDCTimeStamp(ptr->tdcTimeStamp());
  setOrbit(ptr->orbit());
  setChannelId(ptr->channelId());
  setLayerId(ptr->layerId());
  setCameraId(ptr->cameraId());
  setSuperLayerId(ptr->superLayerId());
  setLaterality(ptr->laterality());

  for (int i = 0; i < PAYLOAD_ENTRIES; i++)
    setPayload(ptr->payLoad(i), i);
}

DTPrimitive::DTPrimitive(DTPrimitive* ptr) {
  setTimeCorrection(ptr->timeCorrection());
  setTDCTimeStamp(ptr->tdcTimeStamp());
  setOrbit(ptr->orbit());
  setChannelId(ptr->channelId());
  setLayerId(ptr->layerId());
  setCameraId(ptr->cameraId());
  setSuperLayerId(ptr->superLayerId());
  setLaterality(ptr->laterality());

  for (int i = 0; i < PAYLOAD_ENTRIES; i++)
    setPayload(ptr->payLoad(i), i);
}

DTPrimitive::~DTPrimitive() {}

//------------------------------------------------------------------
//--- Public Methods
//------------------------------------------------------------------
bool DTPrimitive::isValidTime(void) { return (tdcTimeStamp_ >= 0 ? true : false); }

float DTPrimitive::wireHorizPos(void) {
  // For layers with odd-number
  float wireHorizPos = CELL_LENGTH * channelId();
  // If layer is even, you must correct by half a cell
  if (layerId() == 0 || layerId() == 2)
    wireHorizPos += CELL_SEMILENGTH;
  return wireHorizPos;
}
