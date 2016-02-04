// $Id: RBCLogic.h,v 1.2 2009/06/07 21:18:50 aosorio Exp $
#ifndef RBCLOGIC_H 
#define RBCLOGIC_H 1

// Include files
#include "RBCInput.h"
#include "CondFormats/RPCObjects/interface/RBCBoardSpecs.h"

#include <bitset>

/** @class RBCLogic RBCLogic.h
 *  
 *
 *  @author Andres Osorio
 *
 *  email: aosorio@uniandes.edu.co
 *
 *  @date   2008-10-11
 */

class RBCLogic {
public: 
  
  virtual ~RBCLogic() {};
  
  virtual void process ( const RBCInput & , std::bitset<2> & ) = 0;

  virtual void setBoardSpecs( const RBCBoardSpecs::RBCBoardConfig & ) = 0;

  virtual std::bitset<6> * getlayersignal( int ) = 0;
  
  bool m_triggersignal;
  
protected:
  
private:
  
};
#endif // RBCLOGIC_H
