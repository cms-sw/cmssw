// $Id: RPCWheel.h,v 1.4 2009/06/17 15:27:24 aosorio Exp $
#ifndef RPCWHEEL_H 
#define RPCWHEEL_H 1

// Include files
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCEmulator.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/TTUInput.h"
#include "CondFormats/RPCObjects/interface/RBCBoardSpecs.h"

#include<vector>
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
  RPCWheel( );

  virtual ~RPCWheel( ); ///< Destructor
  
  void setProperties( int );
  
  void setProperties( int , const char * );
  
  void setProperties( int , const char *, const char * );
  
  void setSpecifications( const RBCBoardSpecs * );
  
  bool initialise();
  
  void emulate();
  
  bool process( int , const std::map<int,RBCInput*> & );
  
  bool process( int , const std::map<int,TTUInput*> & );
  
  void createWheelMap();
  
  void retrieveWheelMap( TTUInput & );
  
  int  getid() { return m_id; };
  
  void printinfo();

  void print_wheel(const TTUInput & );
  
  std::vector<RBCEmulator*> m_RBCE;
  
protected:
  
private:
  
  int m_id;
  int m_maxrbc;
  int m_maxlayers;
  int m_maxsectors;
  
  std::vector<int> m_sec1id;
  std::vector<int> m_sec2id;
  
  //...

  std::bitset<12>  m_rbcDecision;
  std::bitset<6> * m_wheelmap;

  bool m_debug;
    
};

#endif // RPCWHEEL_H
