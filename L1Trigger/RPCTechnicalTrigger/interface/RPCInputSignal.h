// $Id: RPCInputSignal.h,v 1.1 2009/01/30 15:42:47 aosorio Exp $
#ifndef INTERFACE_RPCINPUTSIGNAL_H 
#define INTERFACE_RPCINPUTSIGNAL_H 1

// Include files

/** @class RPCInputSignal RPCInputSignal.h interface/RPCInputSignal.h
 *  
 *
 *  @author Andres Felipe Osorio Oliveros
 *  @date   2008-11-27
 */
class RPCInputSignal {
public: 
  
  virtual ~RPCInputSignal( ) {}; ///< Destructor

  virtual void clear() = 0;

protected:
  
private:
  
};
#endif // INTERFACE_RPCINPUTSIGNAL_H
