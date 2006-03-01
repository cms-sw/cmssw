#include <iostream>
#include <string>
#include <sstream>
#include <stdlib.h>
#include <cstdlib>
#include <stdexcept>
#include <occi.h>

using namespace std;
using namespace oracle::occi;

#include "OnlineDB/EcalCondDB/interface/EcalDBConnection.h"
#include "OnlineDB/EcalCondDB/interface/DateHandler.h"

EcalDBConnection::EcalDBConnection( string host,
				    string sid,
				    string user,
				    string pass,
				    int port )
  throw(runtime_error)
{
  try {    
    stringstream ss;
    ss << "//" << host << ":" << port << "/" << sid;
    
    env = Environment::createEnvironment(Environment::OBJECT);
    conn = env->createConnection(user, pass, ss.str());
    stmt = conn->createStatement();
  } catch (SQLException &e) {
    throw(runtime_error("ERROR:  Connection Failed:  " + e.getMessage() ));
  }

  this->host = host;
  this->sid = sid;
  this->user = user;
  this->pass = pass;
  this->port = port;
}

EcalDBConnection::~EcalDBConnection() {
  //Close database conection and terminate environment
  try {
    conn->terminateStatement(stmt);
    env->terminateConnection(conn);
    Environment::terminateEnvironment(env);
  } catch (SQLException &e) {
    throw(runtime_error("ERROR:  Destructor Failed:  " + e.getMessage() ));
  }
}
