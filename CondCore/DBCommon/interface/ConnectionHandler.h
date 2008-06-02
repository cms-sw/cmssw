#ifndef CondCore_DBCommon_ConnectionHandler_H
#define CondCore_DBCommon_ConnectionHandler_H
//
// Package:     DBCommon
// Class  :     ConnectionHandler
// 
/**\class ConnectionHandler ConnectionHandler.h CondCore/DBCommon/interface/ConnectionHandler.h
   Top level handler for reigsitration/handling multiple connections in the same session 
   Meyers singleton
*/
//
// Author:      Zhen Xie
//
#include <map>
#include <string>
namespace cond{
  class DBSession;
  class Connection;
  class ConnectionHandler{
  public:
    static ConnectionHandler& Instance();
    /// register connection with a given name and timeout value. timeout value can be -1,0 and n sec
    void registerConnection(const std::string& name,
			    const std::string& con,
			    int timeOutInSec=0);
    /// register userconnection string. This method will translate the userconnect to the real connect using technology proxy. It also initialised the session for the given technology. timeout value can be -1,0 and n sec
    void registerConnection(const std::string& userconnect,
			    cond::DBSession& session,
			    int timeOutInSec=0);
    /// remove connection from connection pool
    void removeConnection( const std::string& name );
    /// global connect
    /// delegate all the registered proxy to connect 
    /// if timeOutInSec !=0, cleanup idle connections with given parameter
    void connect(cond::DBSession* session);
    /// manually disconnect all and clean the registry as well. Otherwise, all connections will be closed and cleaned in the destructor
    void disconnectAll();
    /// contructor
    ConnectionHandler(){}
    /// query connection
    Connection* getConnection( const std::string& name );
  private:
    /// hide copy constructor
    ConnectionHandler( ConnectionHandler& );
    /// hide assign op
    ConnectionHandler& operator=(const ConnectionHandler&);
    /// hide destructor 
    ~ConnectionHandler(); 
  private:
    /// registry of real connection handles
    std::map<std::string,cond::Connection*> m_registry;
  };// class ConnectionHandler
}//ns cond
#endif
