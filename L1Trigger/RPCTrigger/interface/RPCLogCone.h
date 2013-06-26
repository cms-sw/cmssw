#ifndef L1Trigger_RPCLogCone_h
#define L1Trigger_RPCLogCone_h

/** \class RPCLogCone
 *
 * The input for m_PAC. State of strips in smalest unit of volum in RPC trigger
 * system (Logic Cone), defined by 8 strips of reference plane.
 * \author Karol Bunkowski (Warsaw),
 * \author Porting to CMSSW - Tomasz Frueboes
 *
 */

#include <vector>
#include <string>
#include <set>
#include <map>



#include "L1Trigger/RPCTrigger/interface/RPCLogHit.h"

#include "L1Trigger/RPCTrigger/interface/RPCConst.h"
//------------------------------------------------------------------------------

class RPCLogCone {
public:


  /** Hits in one Logic Plane, if strips is fired, its number is added to the map as a key.
    * Vector stores the indexes in DigisVec (stored in L1RpcTrigg) of  Digis that formed log hits
    * Logic m_Strips are diferent from RPC strips - Logic m_Strips it is usaly OR
    * of 2 RPC strips with diferent eta (but the same phi). @see RPCLogHit
    */  
  typedef std::map<int, std::vector<int> > TLogPlane;

  /// Default constructor. No hits, no muon.
  RPCLogCone();

  ///Constructor. Cone coordinates are set.
  RPCLogCone(int m_tower, int logSector, int logSegment);

  ///Constructor. One hit is added, cone coordinates are set from logHit.
  RPCLogCone(const RPCLogHit &logHit);

  ///Constructor. The cone is built from unsigned long long
  RPCLogCone(const unsigned long long &pat, int tower, int logSector, int logSegment);

  /// Compresses cone. Throws exception, if there is more than one hit in any logplane
  unsigned long long getCompressedCone();

  ///Adds next logHit .
  bool addLogHit(const RPCLogHit &logHit);

  TLogPlane getLogPlane(int logPlane) const;

  ///Gets fired strips count in given logPlane.
  int getHitsCnt(int logPlane) const;

  ///Set logic strip as fired. m_digiIdx - index of digi in digis vector stored by L1RpcTrigg
  void setLogStrip(int logPlane, int logStripNum, int m_digiIdx);
  
  ///Set logic strip as fired.
  void setLogStrip(int logPlane, int logStripNum);

  /** Get logic strip state. @return true if fired */
  bool getLogStripState(int logPlane, unsigned int logStripNum) const;

  /** Get vector of didgis indexes (in digis vector stored by L1RpcTrigg) 
    * for given logic strip. If strip was not fired returns empty vector*/
  std::vector<int> getLogStripDigisIdxs(int logPlane, unsigned int logStripNum) const;
  
  void setMuonCode(int code);

  int getMuonCode() const;

  void setMuonSign(int sign);

  int getMuonSign() const;

  ///Changes fired LogStrips: from "stripNum" to "stripNum + pos"
  void shift(int pos);

  bool isPlaneFired(int logPlane) const;

  int getFiredPlanesCnt() const;

  /** @return 0 - trigger not possible, 1 - 3 inner planes fired, 2 - 4 or more planes fired*/
  int possibleTrigger() const;

  int getTower() const;

  int getLogSector() const;

  int getLogSegment() const;

  RPCConst::l1RpcConeCrdnts getConeCrdnts() const;
  
  void setIdx(int index);
  
  int getIdx() const;
  
  std::string toString() const;
  


  
private:
  ///Logic Planes
  std::vector<TLogPlane> m_LogPlanesVec;

  /** Digis that formed log hits in this LogCone, 
    * m_DigisIdx[logPlaneNum][i] gets the index in DigisVec stored in L1RpcTrigg
    */
  std::vector<std::vector<int> > m_DigisIdx;

  RPCConst::l1RpcConeCrdnts m_ConeCrdnts;

  int m_MuonCode;

  int m_MuonSign;

  ///m_Index in LogConesVec stored by L1RpcTrigg
  int m_Index; 
};

typedef std::vector<RPCLogCone> L1RpcLogConesVec;
#endif

