#include <iostream>
#include <string>
#include <sstream>
#include <stdlib.h>
#include <cstdlib>
#include <stdexcept>
#include "OnlineDB/Oracle/interface/Oracle.h"

using namespace std;
using namespace oracle::occi;

#include "OnlineDB/EcalCondDB/interface/EcalDBConnection.h"
#include "OnlineDB/EcalCondDB/interface/DateHandler.h"

EcalDBConnection::EcalDBConnection( string host,
				    string sid,
				    string user,
				    string pass,
				    int port )
  throw(std::runtime_error)
{
    stringstream ss;
  try {    
    ss << "//" << host << ":" << port << "/" << sid;
    
    env = Environment::createEnvironment(Environment::OBJECT);
    conn = env->createConnection(user, pass, ss.str());
    stmt = conn->createStatement();
  } catch (SQLException &e) {
    cout<< ss.str() << endl;
    throw(std::runtime_error("ERROR:  Connection Failed:  " + e.getMessage() ));
  }

  this->host = host;
  this->sid = sid;
  this->user = user;
  this->pass = pass;
  this->port = port;
}

EcalDBConnection::EcalDBConnection( string sid,
				    string user,
				    string pass )
  throw(std::runtime_error)
{
  try {    
    env = Environment::createEnvironment(Environment::OBJECT);
    conn = env->createConnection(user, pass, sid);
    stmt = conn->createStatement();
  } catch (SQLException &e) {
    throw(std::runtime_error("ERROR:  Connection Failed:  " + e.getMessage() ));
  }

  this->host = "";
  this->sid = sid;
  this->user = user;
  this->pass = pass;
  this->port = port;
}

EcalDBConnection::~EcalDBConnection()  throw(std::runtime_error) {
  //Close database conection and terminate environment
  try {
    conn->terminateStatement(stmt);
    env->terminateConnection(conn);
    Environment::terminateEnvironment(env);
  } catch (SQLException &e) {
    throw(std::runtime_error("ERROR:  Destructor Failed:  " + e.getMessage() ));
  }
}
