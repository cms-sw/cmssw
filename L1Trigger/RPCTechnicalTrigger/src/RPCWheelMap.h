// $Id: $
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
  
  void contractMaps();
  
  void prepareData();
  
  int wheelid() { return m_wheelid; };
    
  TTUInput * m_ttuin;

protected:
  
private:
  
  int m_bx;
  int m_wheelid;
  std::bitset<6> * m_wheelmap;
  std::bitset<6> * m_wheelmapbx;
  
  
};
#endif // RPCWHEELMAP_H
