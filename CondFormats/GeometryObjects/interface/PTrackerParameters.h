#ifndef CondFormats_GeometryObjects_PTrackerParameters_h
#define CondFormats_GeometryObjects_PTrackerParameters_h

#include "CondFormats/Serialization/interface/Serializable.h"

class PTrackerParameters
{
 public:
  PTrackerParameters( void ) { } 
  ~PTrackerParameters( void ) { }

  struct PxbItem
  {  
    unsigned int layerStartBit;
    unsigned int ladderStartBit;
    unsigned int moduleStartBit;
    unsigned int layerMask;
    unsigned int ladderMask;
    unsigned int moduleMask;
    
    COND_SERIALIZABLE;
  };

  struct PxfItem
  {
    unsigned int sideStartBit;
    unsigned int diskStartBit;
    unsigned int bladeStartBit;
    unsigned int panelStartBit;
    unsigned int moduleStartBit;
    unsigned int sideMask;
    unsigned int diskMask;
    unsigned int bladeMask;
    unsigned int panelMask;
    unsigned int moduleMask;

    COND_SERIALIZABLE;
  };

  struct TECItem
  {
    unsigned int sideStartBit;
    unsigned int wheelStartBit;
    unsigned int petal_fw_bwStartBit;
    unsigned int petalStartBit;
    unsigned int ringStartBit;
    unsigned int moduleStartBit;
    unsigned int sterStartBit;
    unsigned int sideMask;
    unsigned int wheelMask;
    unsigned int petal_fw_bwMask;
    unsigned int petalMask;
    unsigned int ringMask;
    unsigned int moduleMask;
    unsigned int sterMask;

    COND_SERIALIZABLE;
  };

  struct TIBItem
  {
    unsigned int layerStartBit;
    unsigned int str_fw_bwStartBit;
    unsigned int str_int_extStartBit;
    unsigned int strStartBit;
    unsigned int moduleStartBit;
    unsigned int sterStartBit;
    unsigned int layerMask;
    unsigned int str_fw_bwMask;
    unsigned int str_int_extMask;
    unsigned int strMask;
    unsigned int moduleMask;
    unsigned int sterMask;

    COND_SERIALIZABLE;
  };

  struct TIDItem
  {
    unsigned int sideStartBit;
    unsigned int wheelStartBit;
    unsigned int ringStartBit;
    unsigned int module_fw_bwStartBit;
    unsigned int moduleStartBit;
    unsigned int sterStartBit;
    unsigned int sideMask;
    unsigned int wheelMask;
    unsigned int ringMask;
    unsigned int module_fw_bwMask;
    unsigned int moduleMask;
    unsigned int sterMask;

    COND_SERIALIZABLE;
  };

  struct TOBItem
  {  
    unsigned int layerStartBit;
    unsigned int rod_fw_bwStartBit;
    unsigned int rodStartBit;
    unsigned int moduleStartBit;
    unsigned int sterStartBit;
    unsigned int layerMask;
    unsigned int rod_fw_bwMask;
    unsigned int rodMask;
    unsigned int moduleMask;
    unsigned int sterMask;

    COND_SERIALIZABLE;
  };

  PxbItem pxb;
  PxfItem pxf;
  TECItem tec;
  TIBItem tib;
  TIDItem tid;
  TOBItem tob;

  COND_SERIALIZABLE;
};

#endif
