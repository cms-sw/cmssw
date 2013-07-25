// $Id: RBCLinkBoardGLSignal.h,v 1.1 2009/05/16 19:43:30 aosorio Exp $
#ifndef RBCLINKBOARDGLSIGNAL_H 
#define RBCLINKBOARDGLSIGNAL_H 1

// Include files
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCInput.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/RPCInputSignal.h"

#include <map>

/** @class RBCLinkBoardGLSignal RBCLinkBoardGLSignal.h
 *  
 *
 *  @author Andres Felipe Osorio Oliveros
 *  @date   2008-11-28
 */
class RBCLinkBoardGLSignal : public RPCInputSignal {
public: 
  /// Standard constructor
  RBCLinkBoardGLSignal( ) { }; 
  
  RBCLinkBoardGLSignal( std::map< int, RBCInput* >  * ); 
  
  virtual ~RBCLinkBoardGLSignal( ); ///< Destructor
  
  void clear() { };
  
  std::map< int, RBCInput* >  * m_linkboardin;
  
protected:
  
private:

};
#endif // RBCLINKBOARDGLSIGNAL_H
