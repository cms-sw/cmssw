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
 */

//#include "L1Trigger/RPCTrigger/src/L1RpcParametersDef.h"
#include "L1Trigger/RPCTrigger/src/L1RpcParameters.h"
//------------------------------------------------------------------------------
class L1RpcPacBase {
protected:
  ///Coordinates of LogCone. The coordinates, with which PAC is created - the same as in pac file name
  RPCParam::L1RpcConeCrdnts ConeCrdnts;

  /** Coordinates of current LogCone. The same PAC may be used for several LogCones.
    * @see L1RpcPacManager */
  RPCParam::L1RpcConeCrdnts CurrConeCrdnts;

public:
  //This constructor should be implemented in anly class inherited from L1RpcPacBase.
  //Required in L1RpcPacManager.
  //L1RpcPacXXX(std::string patFilesDir, int tower, int logSector, int logSegment);

  ///Constructor. ConeCrdnts and  CurrConeCrdnts are set.
  L1RpcPacBase(int tower, int logSector, int logSegment) {
    ConeCrdnts.Tower = tower;
    ConeCrdnts.LogSector = logSector;
    ConeCrdnts.LogSegment = logSegment;

    CurrConeCrdnts = ConeCrdnts;
  }

  ///Constructor. ConeCrdnts and  CurrConeCrdnts are set.
  L1RpcPacBase(RPCParam::L1RpcConeCrdnts coneCrdnts): ConeCrdnts(coneCrdnts), CurrConeCrdnts(coneCrdnts) {};

  ///CurrConeCrdnts are set. Called by L1RpcPacManager in GetPac.
  void SetCurrentPosition(int tower, int logSector, int logSegment) {
    CurrConeCrdnts.Tower = tower;
    CurrConeCrdnts.LogSector = logSector;
    CurrConeCrdnts.LogSegment = logSegment;
  };
  
  ///CurrConeCrdnts are set. Called by L1RpcPacManager in GetPac.
  void SetCurrentPosition(RPCParam::L1RpcConeCrdnts coneCrdnts) {
    CurrConeCrdnts = coneCrdnts;
  };

};
#endif




