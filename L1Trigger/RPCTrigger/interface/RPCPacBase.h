#ifndef L1Trigger_RPCPacBase_h
#define L1Trigger_RPCPacBase_h

/** \class RPCPacBase
 *
 * Interface for m_PAC classes. Containes only the coordinates of LogCone,
 * for which given m_PAC works.
 * \author Karol Bunkowski (Warsaw)
 * \note Constructor L1RpcPacXXX(std::string patFilesDir, int m_tower, int logSector, int logSegment) 
 * should be implemented in anly class inherited from RPCPacBase. Required in RPCPacManager.
 * 
 */

#include "L1Trigger/RPCTrigger/interface/RPCConst.h"
//------------------------------------------------------------------------------
class RPCPacBase {

public:
  
  RPCPacBase(int m_tower, int logSector, int logSegment);
  
  RPCPacBase(RPCConst::l1RpcConeCrdnts coneCrdnts);
  
  void setCurrentPosition(int m_tower, int logSector, int logSegment);
    
  void setCurrentPosition(RPCConst::l1RpcConeCrdnts coneCrdnts);
protected:
  ///Coordinates of LogCone.The coordinates, with which m_PAC is created - the same as in pac file name
  RPCConst::l1RpcConeCrdnts m_ConeCrdnts;

  /** Coordinates of current LogCone. The same m_PAC may be used for several LogCones.
    * @see RPCPacManager */
  RPCConst::l1RpcConeCrdnts m_CurrConeCrdnts;

  
};
#endif




