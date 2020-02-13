#include "L1Trigger/DTPhase2Trigger/interface/dtprimitive.h"
#include "L1Trigger/DTPhase2Trigger/interface/constants.h"
#include <iostream> 

//------------------------------------------------------------------
//--- Constructores y destructores
//------------------------------------------------------------------
DTPrimitive::DTPrimitive() {
    //std::cout<<"Creando una 'DTPrimitive'"<<std::endl;
    
    cameraId       = -1;
    superLayerId   = -1;
    layerId        = -1;
    channelId      = -1;
    tdcTimeStamp   = -1;  // Valor negativo => celda sin valor medido
    orbit          = -1;
    timeCorrection = 0;
    laterality     = NONE; 

    for (int i = 0; i < PAYLOAD_ENTRIES; i++) setPayload(0.0, i);
}

DTPrimitive::DTPrimitive(DTPrimitive *ptr) {
    
    //std::cout<<"Clonando una 'DTPrimitive'"<<std::endl;
    
    setTimeOffsetCorrection( ptr->getTimeOffsetCorrection() );
    
    setTDCTime      ( ptr->getTDCTime()      );
    setOrbit        ( ptr->getOrbit()        );
    setChannelId    ( ptr->getChannelId()    );
    setLayerId      ( ptr->getLayerId()      );
    setCameraId     ( ptr->getCameraId()     );
    setSuperLayerId ( ptr->getSuperLayerId() );
    setLaterality   ( ptr->getLaterality()   ); 

    for (int i = 0; i < PAYLOAD_ENTRIES; i++) setPayload(ptr->getPayload(i), i);
}

DTPrimitive::~DTPrimitive() {
    //std::cout<<"Destruyendo una 'DTPrimitive'"<<std::endl;
}

//------------------------------------------------------------------
//--- Métodos públicos
//------------------------------------------------------------------
bool DTPrimitive::isValidTime(void) {
    return (tdcTimeStamp >= 0 ? true : false);
}

//------------------------------------------------------------------
//--- Métodos get / set
//------------------------------------------------------------------
int DTPrimitive::getTimeOffsetCorrection(void) { return timeCorrection; }
void DTPrimitive::setTimeOffsetCorrection(int time) { timeCorrection = time; }

/*
 * TDC Time es el tiempo "en bruto" medido por el TDC, pero convertido a un
 * escalar único en 'ns'. Incluye las demoras por la electrónica y el desfase
 * por el Bunch_0
 */

int  DTPrimitive::getTDCTime(void)         { return tdcTimeStamp;   }
void DTPrimitive::setTDCTime(int tstamp)   { tdcTimeStamp = tstamp; }
int  DTPrimitive::getTDCTimeNoOffset(void) {
    return tdcTimeStamp - timeCorrection ;
}

int  DTPrimitive::getOrbit(void)    { return orbit; }
void DTPrimitive::setOrbit(int orb) { orbit = orb;  }

int DTPrimitive::getTDCCoarsePart(void) {
    return ( ((tdcTimeStamp && TDC_TIME_COARSE_MASK) >> TDC_TIME_FINE_MASK) );
}

int DTPrimitive::getTDCFinePart(void) {
    return ( (tdcTimeStamp && TDC_TIME_FINE_MASK) );
}

double DTPrimitive::getPayload(int idx) { return hitTag[idx]; }
void DTPrimitive::setPayload(double hitTag, int idx) {
    this->hitTag[idx] = hitTag;
}

int  DTPrimitive::getChannelId(void)        { return channelId;    }
void DTPrimitive::setChannelId(int channel) { channelId = channel; }
int  DTPrimitive::getLayerId(void)          { return layerId;      }
void DTPrimitive::setLayerId(int layer)     { layerId = layer;     }
int  DTPrimitive::getCameraId(void)         { return cameraId;     }
void DTPrimitive::setCameraId(int camera)   { cameraId = camera;   }
int  DTPrimitive::getSuperLayerId(void)     { return superLayerId; }
void DTPrimitive::setSuperLayerId(int lay)  { superLayerId = lay;  }

float DTPrimitive::getWireHorizPos(void) {
    // Para layers con número impar.
    float wireHorizPos = CELL_LENGTH * getChannelId();
    // Si la layer es par, hay que corregir media semi-celda.
    if (getLayerId() == 0 || getLayerId() == 2) wireHorizPos += CELL_SEMILENGTH;
    return wireHorizPos;
}


