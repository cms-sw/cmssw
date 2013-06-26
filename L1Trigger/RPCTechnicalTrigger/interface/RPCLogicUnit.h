// $Id: RPCLogicUnit.h,v 1.2 2009/08/09 11:11:36 aosorio Exp $
#ifndef INTERFACE_RPCLOGICUNIT_H 
#define INTERFACE_RPCLOGICUNIT_H 1

// Include files

/** @class RPCLogicUnit RPCLogicUnit.h interface/RPCLogicUnit.h
 *  
 *  utilitary class: not fully exploited yet 
 *  
 *  @author Andres Osorio
 *  @date   2008-10-25
 */
class RPCLogicUnit {
public: 
  /// Standard constructor
  RPCLogicUnit( ) {};
  
  RPCLogicUnit( int, int, int );
    
  virtual ~RPCLogicUnit( ); ///< Destructor
  
  int m_propA;
  int m_propB;
  int m_propC;

protected:
  
private:
  
};
#endif // INTERFACE_RPCLOGICUNIT_H
