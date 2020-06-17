#include "L1Trigger/DTTriggerPhase2/interface/DTprimitive.h"
#include "L1Trigger/DTTriggerPhase2/interface/constants.h"
#include <iostream>
#include <iomanip>

//------------------------------------------------------------------
//--- Constructores y destructores
//------------------------------------------------------------------
DTPrimitive::DTPrimitive() {
  //std::cout<<"Creando una 'DTPrimitive'"<<std::endl;

  cameraId_ = -1;
  superLayerId_ = -1;
  layerId_ = -1;
  channelId_ = -1;
  tdcTimeStamp_ = -1;  // Valor negativo => celda sin valor medido
  orbit_ = -1;
  timeCorrection_ = 0;
  laterality_ = NONE;

  for (int i = 0; i < PAYLOAD_ENTRIES; i++)
    setPayload(0.0, i);
}

DTPrimitive::DTPrimitive(DTPrimitivePtr &ptr) {
  //std::cout<<"Clonando una 'DTPrimitive'"<<std::endl;

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
  //std::cout<<"Clonando una 'DTPrimitive'"<<std::endl;

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

DTPrimitive::~DTPrimitive() {
  //std::cout<<"Destruyendo una 'DTPrimitive'"<<std::endl;
}

//------------------------------------------------------------------
//--- Métodos públicos
//------------------------------------------------------------------
bool DTPrimitive::isValidTime(void) { return (tdcTimeStamp_ >= 0 ? true : false); }

float DTPrimitive::wireHorizPos(void) {
  // Para layers con número impar.
  float wireHorizPos = CELL_LENGTH * channelId();
  // Si la layer es par, hay que corregir media semi-celda.
  if (layerId() == 0 || layerId() == 2)
    wireHorizPos += CELL_SEMILENGTH;
  return wireHorizPos;
}

