#ifndef CondCore_DBCommon_ConnectionHandler_H
#define CondCore_DBCommon_ConnectionHandler_H
#include <vector>
#include <string>
namespace cond{
  class DBSession;
  class Connection;
  /*
    handle connections registered in the same session and connection timeout 
    Meyers singleton
    user level interface
  **/ 
  class ConnectionHandler{
  public:
    static ConnectionHandler& Instance();
    /// register conditions connection with pool capability
    void registerConnection(const std::string& con,
			    const std::string& filecatalog,
			    unsigned int timeOutInSec=0);
    /// register coral only connection
    void registerConnection(const std::string& con,
			    unsigned int timeOutInSec=0);
    /// global connect
    /// delegate all the registered proxy to connect 
    /// if timeOutInSec !=0, cleanup idle connections with given parameter
    void connect(cond::DBSession* session);
    /// global disconnect
    void disconnectAll();
    /// contructor
    ConnectionHandler(){}
  private:
    /// hide copy constructor
    ConnectionHandler( ConnectionHandler& );
    /// hide assign op
    ConnectionHandler& operator=(const ConnectionHandler&);
    /// hide destructor 
    ~ConnectionHandler(); 
  private:
    /// registry of real connection handles
    std::vector<cond::Connection*> m_registry;
  };// class ConnectionHandler
}//ns cond
#endif
