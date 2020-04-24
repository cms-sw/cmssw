//-------------------------------------------------
//
//   \class L1MuGMTChannelMask
//
/**   Description:   GMT input channel mask
 *                  
*/                  
//
//   $Date$
//   $Revision$
//
//
//   Author :
//   Ivan Mikulec      HEPHY / Vienna
//
//
//--------------------------------------------------
#ifndef CondFormatsL1TObjects_L1MuGMTChannelMask_h
#define CondFormatsL1TObjects_L1MuGMTChannelMask_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <string>

class L1MuGMTChannelMask {
public:
  L1MuGMTChannelMask() {}
  ~L1MuGMTChannelMask() {}
  
  void setSubsystemMask(const unsigned SubsystemMask) { m_SubsystemMask = SubsystemMask; }
  unsigned getSubsystemMask() const { return m_SubsystemMask; }

private:

  unsigned m_SubsystemMask;


  COND_SERIALIZABLE;
};


#endif

