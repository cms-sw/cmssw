/*******************************************************************************
*                                                                              *
*  Karol Bunkowski                                                             *
*  Warsaw University 2004                                                      *
*                                                                              *
*******************************************************************************/
#ifndef L1RpcPacBaseH
#define L1RpcPacBaseH

/** \class RPCPacBase
 *
 * Interface for m_PAC classes. Containes only the coordinates of LogCone,
 * for which given m_PAC works.
 * \author Karol Bunkowski (Warsaw)
 * \note Constructor L1RpcPacXXX(std::string patFilesDir, int m_tower, int logSector, int logSegment) 
 * should be implemented in anly class inherited from RPCPacBase. Required in RPCPacManager.
 * 
 */

#include "L1Trigger/RPCTrigger/src/RPCConst.h"
//------------------------------------------------------------------------------
class RPCPacBase {
protected:
  ///Coordinates of LogCone. The coordinates, with which m_PAC is created - the same as in pac file name
  RPCConst::l1RpcConeCrdnts m_ConeCrdnts;

  /** Coordinates of current LogCone. The same m_PAC may be used for several LogCones.
    * @see RPCPacManager */
  RPCConst::l1RpcConeCrdnts m_CurrConeCrdnts;

public:
  
  RPCPacBase(int m_tower, int logSector, int logSegment);
  
  RPCPacBase(RPCConst::l1RpcConeCrdnts coneCrdnts);
  
  void setCurrentPosition(int m_tower, int logSector, int logSegment);
    
  void setCurrentPosition(RPCConst::l1RpcConeCrdnts coneCrdnts);

};
#endif




