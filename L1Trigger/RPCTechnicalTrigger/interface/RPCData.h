// $Id: $
#ifndef INTERFACE_RPCDATA_H 
#define INTERFACE_RPCDATA_H 1

// Include files
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCInput.h"

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <ios>
#include <cmath>

/** @class RPCData RPCData.h interface/RPCData.h
 *  
 *
 *  Data structure consisting of wheel + sector + ORs signals
 *
 *  @author Andres Osorio
 *  @date   2008-11-18
 */
class RPCData {
public: 
  /// Standard constructor
  RPCData( ); 
  virtual ~RPCData( ); ///< Destructor

  int         m_wheel;
  int      *  m_sec1;
  int      *  m_sec2;
  RBCInput * m_orsignals;

  friend std::istream& operator>>(std::istream &, RPCData &);
  friend std::ostream& operator<<(std::ostream &, RPCData &);
  
protected:

private:

};
#endif // INTERFACE_RPCDATA_H
