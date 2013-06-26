// $Id: RPCData.h,v 1.4 2013/03/20 15:45:25 wdd Exp $
#ifndef INTERFACE_RPCDATA_H 
#define INTERFACE_RPCDATA_H 1

// Include files
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCInput.h"

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <ios>
#include <cmath>
#include <map>

/** @class RPCData RPCData.h interface/RPCData.h
 *  
 *
 *  Data structure consisting of wheel + sector + ORs signals
 *
 *  @author Andres Osorio
 *  @date   2008-11-18
 */

namespace l1trigger {
  class Counters {
  public:
  
    Counters() {};
    Counters( int );
  
    virtual ~Counters();
  
    void incrementSector( int );
  
    void printSummary();
  
    void evalCounters();
  
    int m_wheelid;
    int m_nearSide;
    int m_farSide;
    int m_wheel;
    std::map<int,int> m_sector;
  };
}

class RPCData {
public: 
  /// Standard constructor
  RPCData( ); 
  virtual ~RPCData( ); ///< Destructor

  int         m_wheel;
  int      *  m_sec1;
  int      *  m_sec2;
  RBCInput *  m_orsignals;

  friend std::istream& operator>>(std::istream &, RPCData &);
  friend std::ostream& operator<<(std::ostream &, RPCData &);
  
  int wheelIdx() //wheel index starts from 0
  {
    return (m_wheel + 2);
  }
  
protected:
  
private:

};
#endif // INTERFACE_RPCDATA_H
