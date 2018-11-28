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
private:
  static constexpr int m_maxBx = 7;
  static constexpr int m_maxSectors = 12;
  static constexpr int m_maxBxWindow = 3; //... considering that we have a bxing in the range [-3,+3]
  
public: 
 
  RPCWheelMap( int );
  
  void addHit( int , int , int );
  
  void prepareData();
  
  int wheelid() const { return m_wheelid; };
  
  int wheelIdx() const { return (m_wheelid+2); };
  
  std::array<TTUInput,m_maxBx> m_ttuinVec;
  
protected:
  
private:
  int m_bx;
  int m_wheelid;
  
  std::array<std::bitset<6>,m_maxSectors>  m_wheelMap;
  std::array<std::bitset<6>,m_maxSectors * m_maxBx> m_wheelMapBx;
  
  bool m_debug;
  
};
#endif // RPCWHEELMAP_H
