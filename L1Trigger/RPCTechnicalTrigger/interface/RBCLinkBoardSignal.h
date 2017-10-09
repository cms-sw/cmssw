#ifndef RBCLINKBOARDSIGNAL_H 
#define RBCLINKBOARDSIGNAL_H 1

// Include files
#include "L1Trigger/RPCTechnicalTrigger/interface/RPCInputSignal.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCInput.h"

/** @class RBCLinkBoardSignal RBCLinkBoardSignal.h
 *  
 *
 *  @author Andres Felipe Osorio Oliveros
 *  @date   2008-11-27
 */
class RBCLinkBoardSignal : public RPCInputSignal {
public: 
  /// Standard constructor
  RBCLinkBoardSignal( ) {}; 

  RBCLinkBoardSignal( RBCInput * ); 
  
  ~RBCLinkBoardSignal( ) override; ///< Destructor
  
  void clear() override { };

  RBCInput * m_linkboardin;
  
protected:

private:

};
#endif // RBCLINKBOARDSIGNAL_H
