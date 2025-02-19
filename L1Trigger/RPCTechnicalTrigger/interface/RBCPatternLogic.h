// $Id: RBCPatternLogic.h,v 1.2 2009/06/07 21:18:50 aosorio Exp $
#ifndef RBCPATTERNLOGIC_H 
#define RBCPATTERNLOGIC_H 1

// Include files
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCLogic.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCInput.h"

#include <iostream>

/** @class RBCPatternLogic RBCPatternLogic.h
 *  
 *
 *  @author Andres Osorio
 *
 *  email: aosorio@uniandes.edu.co
 *
 *  @date   2008-10-15
 */
class RBCPatternLogic : public RBCLogic {
public: 
  /// Standard constructor
  RBCPatternLogic( ); 
  
  virtual ~RBCPatternLogic( ); ///< Destructor
  
  void process(const RBCInput & , std::bitset<2> & );

  void setBoardSpecs( const RBCBoardSpecs::RBCBoardConfig & );

  std::bitset<6> * getlayersignal( int _idx ) { return &m_testlayer[_idx];};
  
protected:
  
private:
  
  std::bitset<6> m_testlayer[2];
  
};
#endif // RBCPATTERNLOGIC_H
