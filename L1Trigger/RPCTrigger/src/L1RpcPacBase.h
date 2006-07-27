/*******************************************************************************
*                                                                              *
*  Karol Bunkowski                                                             *
*  Warsaw University 2004                                                      *
*                                                                              *
*******************************************************************************/
#ifndef L1RpcPacBaseH
#define L1RpcPacBaseH

/** \class L1RpcPacBase
 *
 * Interface for PAC classes. Containes only the coordinates of LogCone,
 * for which given PAC works.
 * \author Karol Bunkowski (Warsaw)
 * \note Constructor L1RpcPacXXX(std::string patFilesDir, int tower, int logSector, int logSegment) 
 * should be implemented in anly class inherited from L1RpcPacBase. Required in L1RpcPacManager.
 * 
 */

//#include "L1Trigger/RPCTrigger/src/L1RpcParametersDef.h"
#include "L1Trigger/RPCTrigger/src/L1RpcParameters.h"
//------------------------------------------------------------------------------
class L1RpcPacBase {
protected:
  ///Coordinates of LogCone. The coordinates, with which PAC is created - the same as in pac file name
  rpcparam::L1RpcConeCrdnts ConeCrdnts;

  /** Coordinates of current LogCone. The same PAC may be used for several LogCones.
    * @see L1RpcPacManager */
  rpcparam::L1RpcConeCrdnts CurrConeCrdnts;

public:
  
  L1RpcPacBase(int tower, int logSector, int logSegment);
  
  L1RpcPacBase(rpcparam::L1RpcConeCrdnts coneCrdnts);
  
  void SetCurrentPosition(int tower, int logSector, int logSegment);
    
  void SetCurrentPosition(rpcparam::L1RpcConeCrdnts coneCrdnts);

};
#endif




