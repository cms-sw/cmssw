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
  RPCWheel(RPCWheel&&) = default;
  RPCWheel& operator=(RPCWheel&&) = default;
  
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
  
  int  getid() const { return m_id; };
  
  void printinfo() const;

  void print_wheel(const TTUInput & ) const;
  
  
protected:
  
private:
  std::vector<std::unique_ptr<RBCEmulator>> m_RBCE;
  
  int m_id;
  static constexpr int m_maxrbc = 6;
  static constexpr int m_maxlayers = 6;
  static constexpr int m_maxsectors = 12;
  
  //...

  std::bitset<12>  m_rbcDecision;
  std::array<std::bitset<6>,12> m_wheelmap;

  bool m_debug;
    
};

#endif // RPCWHEEL_H
