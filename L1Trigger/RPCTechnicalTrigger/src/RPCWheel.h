// $Id: RPCWheel.h,v 1.1 2009/02/05 13:46:21 aosorio Exp $
#ifndef RPCWHEEL_H 
#define RPCWHEEL_H 1

// Include files
#include "L1Trigger/RPCTechnicalTrigger/src/RBCEmulator.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/TTUInput.h"
#include "CondFormats/RPCObjects/interface/RBCBoardSpecs.h"

#include<map>

/** @class RPCWheel RPCWheel.h
 *  
 *
 *  @author Andres Osorio
 *
 *  email: aosorio@uniandes.edu.co
 *
 *  @date   2008-10-15
 */
void print_wheel( const TTUInput & );

class RPCWheel {
public: 
  /// Standard constructor
  RPCWheel( ) { };

  RPCWheel( int );
  
  RPCWheel( int , const char * );

  RPCWheel( int , const char *, const char * );
  
  virtual ~RPCWheel( ); ///< Destructor

  void setSpecifications( const RBCBoardSpecs * );
  
  bool initialise();
  
  void emulate();
  
  bool process( const std::map<int,RBCInput*> & );
  
  bool process( const std::map<int,TTUInput*> & );
  
  void createWheelMap();
  
  void retrieveWheelMap( TTUInput & );
  
  int  getid() { return m_id; };
  
  void printinfo();

  RBCEmulator * m_RBCE[6];
  
protected:
  
private:
  
  int m_id;
  
  //...
  std::bitset<6> * m_wheelmap[12];

  bool m_debug;
    
};

#endif // RPCWHEEL_H
