#include "CondCore/CondDB/interface/ConnectionPool.h"

#include "CondCore/Utilities/interface/Utilities.h"
#include "CondCore/Utilities/interface/CondDBImport.h"
#include <iostream>

#include <sstream>

namespace cond {

  class TestIovUtilities : public cond::Utilities {
    public:
      TestIovUtilities();
      ~TestIovUtilities();
      int execute();
  };
}

cond::TestIovUtilities::TestIovUtilities():Utilities("conddb_copy_iov"){
  addConnectOption("connect","c","target connection string (required)");
  addAuthenticationOptions();
  addOption<bool>("read","r","de-serialize the specified payload");
  addOption<std::string>("hash","x","hash of the payload");
}

cond::TestIovUtilities::~TestIovUtilities(){
}

int cond::TestIovUtilities::execute(){

  bool debug = hasDebug();
  std::string connect = getOptionValue<std::string>("connect");

  std::string hash("");
  bool read = hasOptionValue( "read" );
  if( read ){
    hash = getOptionValue<std::string>("hash");
  }

  persistency::ConnectionPool connPool;
  if( hasOptionValue("authPath") ){
    connPool.setAuthenticationPath( getOptionValue<std::string>( "authPath") ); 
  }
  connPool.configure();

  std::cout <<"# Connecting to source database on "<<connect<<std::endl;
  persistency::Session session = connPool.createSession( connect );

  session.transaction().start( true );
  try{
    persistency::fetch( hash, session ); 
    std::cout <<"De-serialization of payload "<<hash<<" performed correctly."<<std::endl;
  } catch ( const persistency::Exception& e ){
    std::cout <<"De-serialization of payload "<<hash<<" failed: "<<e.what()<<std::endl; 
  }
  session.transaction().commit();
  return 0;
}

int main( int argc, char** argv ){

  cond::TestIovUtilities utilities;
  return utilities.run(argc,argv);
}

