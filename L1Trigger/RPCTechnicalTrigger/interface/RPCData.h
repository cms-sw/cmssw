#ifndef INTERFACE_RPCDATA_H 
#define INTERFACE_RPCDATA_H 1

// Include files
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCInput.h"

#include <cstdlib>
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
  
    explicit Counters( int );
  
    void incrementSector( int );
  
    void printSummary() const;
  
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
  ~RPCData( ) = default; ///< Destructor

  int         m_wheel;
  std::array<int,6>  m_sec1;
  std::array<int ,6> m_sec2;
  std::array<RBCInput,6>  m_orsignals;

  friend std::istream& operator>>(std::istream &, RPCData &);
  friend std::ostream& operator<<(std::ostream &, RPCData const &);
  
  int wheelIdx() const //wheel index starts from 0
  {
    return (m_wheel + 2);
  }
  
protected:
  
private:

};
#endif // INTERFACE_RPCDATA_H
