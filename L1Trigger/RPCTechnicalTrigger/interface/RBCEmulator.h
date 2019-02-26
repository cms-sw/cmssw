#ifndef RBCEMULATOR_H 
#define RBCEMULATOR_H 1

// Include files
#include "L1Trigger/RPCTechnicalTrigger/interface/LogicTool.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCId.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCInput.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCConfiguration.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/ProcessInputSignal.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/RPCInputSignal.h"

#include "CondFormats/RPCObjects/interface/RBCBoardSpecs.h"

#include <memory>
/** @class RBCEmulator RBCEmulator.h
 *  
 *
 *  @author Andres Osorio, Flavio Loddo, Marcello Maggi
 *
 *  email: aosorio@uniandes.edu.co
 *
 *  @date   2008-10-10
 */

class RBCEmulator {
public: 
  /// Standard constructor
  RBCEmulator( ); 
  
  RBCEmulator( const char * ); 

  RBCEmulator( const char * , const char * ); 
  
  RBCEmulator( const char * , const char * , int, int *); 
  
  void setSpecifications( const RBCBoardSpecs * );
    
  bool initialise();
  
  void setid( int , int * );
  
  void emulate();

  void emulate( RBCInput * );

  void reset();
  
  std::bitset<6> * getlayersignal( int idx ) { return m_layersignal[idx];};

  bool getdecision( int idx ) const { return m_decision[idx];};
    
  void printinfo() const;
  
  void printlayerinfo() const;
  
  const RBCId& rbcinfo() const { return m_rbcinfo;}
  
protected:
  
private:
  RBCId           m_rbcinfo;
  
  std::unique_ptr<ProcessInputSignal> m_signal;
  
  std::unique_ptr<RBCConfiguration>   m_rbcconf;
  
  RBCInput           m_input;
  
  std::bitset<6> * m_layersignal[2];
  
  std::bitset<2> m_decision;
  
  std::array<std::bitset<6>,2> m_layersignalVec;
  
  //...
  std::string m_logtype;

  bool m_debug;
    
};
#endif // RBCEMULATOR_H
