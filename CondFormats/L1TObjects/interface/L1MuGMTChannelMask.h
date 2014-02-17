//-------------------------------------------------
//
//   \class L1MuGMTChannelMask
//
/**   Description:   GMT input channel mask
 *                  
*/                  
//
//   $Date: 2008/11/05 17:19:49 $
//   $Revision: 1.1 $
//
//
//   Author :
//   Ivan Mikulec      HEPHY / Vienna
//
//
//--------------------------------------------------
#ifndef CondFormatsL1TObjects_L1MuGMTChannelMask_h
#define CondFormatsL1TObjects_L1MuGMTChannelMask_h

#include <string>

class L1MuGMTChannelMask {
public:
  L1MuGMTChannelMask() {}
  ~L1MuGMTChannelMask() {}
  
  void setSubsystemMask(const unsigned SubsystemMask) { m_SubsystemMask = SubsystemMask; }
  unsigned getSubsystemMask() const { return m_SubsystemMask; }

private:

  unsigned m_SubsystemMask;

};


#endif

