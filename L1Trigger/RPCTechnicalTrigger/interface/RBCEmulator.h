// $Id: RBCEmulator.h,v 1.9 2009/06/17 15:27:23 aosorio Exp $
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
  
  virtual ~RBCEmulator( ); ///< Destructor

  void setSpecifications( const RBCBoardSpecs * );
    
  bool initialise();
  
  void setid( int , int * );
  
  void emulate();

  void emulate( RBCInput * );

  void reset();
  
  std::bitset<6> * getlayersignal( int idx ) { return m_layersignal[idx];};

  bool getdecision( int idx ) { return m_decision[idx];};
    
  void printinfo();
  
  void printlayerinfo();
  
  RBCId          * m_rbcinfo;
  
protected:
  
private:
  
  ProcessInputSignal * m_signal;
  
  RBCConfiguration   * m_rbcconf;
  
  RBCInput           * m_input;
  
  std::bitset<6> * m_layersignal[2];
  
  std::bitset<2> m_decision;
  
  std::vector< std::bitset<6> *> m_layersignalVec;
  
  //...
  
  int m_bx;
  
  std::string m_logtype;

  bool m_debug;
    
};
#endif // RBCEMULATOR_H
