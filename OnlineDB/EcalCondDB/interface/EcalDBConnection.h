#ifndef ECALDBCONNECTION_HH
#define ECALDBCONNECTION_HH

#include "OnlineDB/Oracle/interface/Oracle.h"
#include <string>
#include <stdexcept>
#include <iostream>

#include "OnlineDB/EcalCondDB/interface/DateHandler.h"


/**
 *  A wrapper class for the oracle Connection and Environment classes
 *  along with any OCCI-dependent methods that are useful for any database
 */
class EcalDBConnection {
 public:

  /******************\
  -  public methods  -
  \******************/

  /**
   *  Constructor
   *  Makes a connection to an oracle database without TNS_ADMIN
   */
  EcalDBConnection( std::string host,
		    std::string sid,
		    std::string user,
		    std::string pass,
		    int port = 1521 )
    throw(std::runtime_error);



  /**
   *  Constructor
   *  Makes a connection to an oracle database using TNS_ADMIN
   */
  EcalDBConnection( std::string sid,
		    std::string user,
		    std::string pass )
    throw(std::runtime_error);
  


  /**
   *  Destructor
   */
  virtual ~EcalDBConnection() throw(std::runtime_error);

  /**
   *  Get a new Statement
   */
  inline oracle::occi::Statement* createStatement()
    {
      std::cout << "Creating statement" << std::endl;
      return conn->createStatement();
    }

  inline void terminateStatement(oracle::occi::Statement* stmt)
    {
      std::cout << "Creating statement" << std::endl;
      conn->terminateStatement(stmt);
    }

  /** 
   *  Tranaction manaagement
   */
  inline void commit() { conn->commit(); }
  inline void rollback() { conn->rollback(); }

  /**
   *  Get a new clob locator
   */
  inline oracle::occi::Clob getClobLocator()
    {
      return oracle::occi::Clob(conn);
    }

  oracle::occi::Environment* getEnv() const { return env; };
  oracle::occi::Connection* getConn() const { return conn; };

 protected:
  
  /***********************\
  -  protected variables  -
  \***********************/

  oracle::occi::Environment* env;
  oracle::occi::Connection* conn;
  oracle::occi::Statement* stmt;

  std::string host;
  int port;
  std::string sid;
  std::string user;
  std::string pass;

};

#endif
