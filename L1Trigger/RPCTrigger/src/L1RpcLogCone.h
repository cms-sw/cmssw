/*******************************************************************************
*                                                                              *
*  Karol Bunkowski                                                             *
*  Warsaw University 2002                                                      *
*                                                                              *
*******************************************************************************/

#ifndef L1RpcLogConeH
#define L1RpcLogConeH

/** \class L1RpcLogCone
 *
 * The input for PAC. State of strips in smalest unit of volum in RPC trigger
 * system (Logic Cone), defined by 8 strips of reference plane.
 * \author Karol Bunkowski (Warsaw),
 * \author Porting to CMSSW - Tomasz Frueboes
 *
 */

#include <vector>
#include <string>
#include <set>
#include <map>

#include "L1Trigger/RPCTrigger/src/L1RpcLogHit.h"

#include "L1Trigger/RPCTrigger/src/L1RpcParameters.h"
//------------------------------------------------------------------------------

class L1RpcLogCone {
public:


  /** Hits in one Logic Plane, if strips is fired, its number is added to the map as a key.
    * Vector stores the indexes in DigisVec (stored in L1RpcTrigg) of  Digis that formed log hits
    * Logic Strips are diferent from RPC strips - Logic Strips it is usaly OR
    * of 2 RPC strips with diferent eta (but the same phi). @see L1RpcLogHit
    */  
  typedef std::map<int, std::vector<int> > TLogPlane;
private:
  ///Logic Planes
  std::vector<TLogPlane> LogPlanesVec;

  /** Digis that formed log hits in this LogCone, 
    * DigisIdx[logPlaneNum][i] gets the index in DigisVec stored in L1RpcTrigg
    */
  std::vector<std::vector<int> > DigisIdx;

  RPCParam::L1RpcConeCrdnts ConeCrdnts;

  int MuonCode;

  int MuonSign;

  ///Index in LogConesVec stored by L1RpcTrigg
  int Index; 
public:

  /// Default constructor. No hits, no muon.
  L1RpcLogCone();

  ///Constructor. Cone coordinates are set.
  L1RpcLogCone(int tower, int logSector, int logSegment);
  
  ///Constructor. One hit is added, cone coordinates are set from logHit.
  L1RpcLogCone(const L1RpcLogHit &logHit);

  ///Adds next logHit .
  bool AddLogHit(const L1RpcLogHit &logHit);

  TLogPlane GetLogPlane(int logPlane) const;

  ///Gets fired strips count in given logPlane.
  int GetHitsCnt(int logPlane) const;

  ///Set logic strip as fired. digiIdx - index of digi in digis vector stored by L1RpcTrigg
  void SetLogStrip(int logPlane, int logStripNum, int digiIdx);
  
  ///Set logic strip as fired.
  void SetLogStrip(int logPlane, int logStripNum);

  /** Get logic strip state. @return true if fired */
  bool GetLogStripState(int logPlane, unsigned int logStripNum) const;

  /** Get vector of didgis indexes (in digis vector stored by L1RpcTrigg) 
    * for given logic strip. If strip was not fired returns empty vector*/
  std::vector<int> GetLogStripDigisIdxs(int logPlane, unsigned int logStripNum) const;
  
  void SetMuonCode(int code);

  int GetMuonCode() const;

  void SetMuonSign(int sign);

  int GetMuonSign() const;

  ///Changes fired LogStrips: from "stripNum" to "stripNum + pos"
  void Shift(int pos);

  bool IsPlaneFired(int logPlane) const;

  int GetFiredPlanesCnt() const;

  /** @return 0 - trigger not possible, 1 - 3 inner planes fired, 2 - 4 or more planes fired*/
  int PossibleTrigger() const;

  int GetTower() const;

  int GetLogSector() const;

  int GetLogSegment() const;

  RPCParam::L1RpcConeCrdnts GetConeCrdnts() const;
  

  void SetIdx(int index);
	
  int GetIdx() const;
  
};

typedef std::vector<L1RpcLogCone> L1RpcLogConesVec;
#endif

