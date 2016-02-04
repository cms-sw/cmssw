// $Id: RPCWheelMap.h,v 1.3 2009/05/26 17:40:37 aosorio Exp $
#ifndef RPCWHEELMAP_H 
#define RPCWHEELMAP_H 1

// Include files
#include "L1Trigger/RPCTechnicalTrigger/interface/TTUInput.h"
#include <bitset>

/** @class RPCWheelMap RPCWheelMap.h
 *  
 *
 *  @author Andres Felipe Osorio Oliveros
 *  @date   2008-11-24
 */
class RPCWheelMap {
public: 
  /// Standard constructor
  RPCWheelMap( ) {};

  RPCWheelMap( int );
  
  virtual ~RPCWheelMap( ); ///< Destructor
  
  void addHit( int , int , int );
  
  void prepareData();
  
  int wheelid() { return m_wheelid; };
  
  int wheelIdx() { return (m_wheelid+2); };
  
  TTUInput * m_ttuinVec;
  
protected:
  
private:
  
  int m_bx;
  int m_wheelid;
  int m_maxBx;
  int m_maxSectors;
  int m_maxBxWindow;
  
  std::bitset<6> * m_wheelMap;
  std::bitset<6> * m_wheelMapBx;
  
  bool m_debug;
  
};
#endif // RPCWHEELMAP_H
