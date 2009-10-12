// $Id: $
#ifndef INTERFACE_TTULOGIC_H 
#define INTERFACE_TTULOGIC_H 1

// Include files
#include "L1Trigger/RPCTechnicalTrigger/interface/TTUInput.h"



/** @class TTULogic TTULogic.h interface/TTULogic.h
 *  
 *
 *  @author Andres Osorio
 * 
 *  email: aosorio@uniandes.edu.co
 *
 *  @date   2008-10-16
 */

class TTULogic {
public: 
  
  virtual ~TTULogic( ){}; 
  
  virtual bool process ( const TTUInput & ) = 0;
  
  bool m_triggersignal;

protected:
  
private:
  
};
#endif // INTERFACE_TTULOGIC_H
