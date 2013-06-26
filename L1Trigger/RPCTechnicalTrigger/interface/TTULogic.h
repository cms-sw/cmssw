// $Id: TTULogic.h,v 1.3 2009/08/09 11:11:36 aosorio Exp $
#ifndef INTERFACE_TTULOGIC_H 
#define INTERFACE_TTULOGIC_H 1

// Include files
#include "L1Trigger/RPCTechnicalTrigger/interface/TTUInput.h"
#include "CondFormats/RPCObjects/interface/TTUBoardSpecs.h"


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

  virtual void setBoardSpecs( const TTUBoardSpecs::TTUBoardConfig & ) = 0;
  
  virtual void setOption( int option ) {
    m_option = option;
  };
    
  int m_option;
  
  bool m_triggersignal;

protected:
  
private:
  
};
#endif // INTERFACE_TTULOGIC_H
