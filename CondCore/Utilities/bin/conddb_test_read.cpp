#include "CondCore/CondDB/interface/ConnectionPool.h"

#include "CondCore/Utilities/interface/Utilities.h"
#include "CondCore/Utilities/interface/CondDBImport.h"
#include <iostream>

#include <sstream>
#include <boost/tokenizer.hpp>

namespace cond {

  class TestReadUtilities : public cond::Utilities {
    public:
      TestReadUtilities();
      ~TestReadUtilities();
      int execute();
  };
}

cond::TestReadUtilities::TestReadUtilities():Utilities("conddb_copy_iov"){
  addConnectOption("connect","c","target connection string (required)");
  addAuthenticationOptions();
  //addOption<bool>("deserialize","d","de-serialize the specified payload");
  addOption<std::string>("hashes","x","space-separated list of hashes of the payloads");
  addOption<std::string>("tag","t","tag for the iov-based search");
  addOption<std::string>("iovs","i","space-separated list of target times ( run, lumi or timestamp)");
  addOption<bool>("run","r","run-labeled transaction");
}

cond::TestReadUtilities::~TestReadUtilities(){
}

int cond::TestReadUtilities::execute(){

  bool debug = hasDebug();
  std::string connect = getOptionValue<std::string>("connect");

  typedef boost::tokenizer<boost::char_separator<char>> tokenizer;
  std::vector<std::string> hashes;
  std::string tag("");
  std::vector<cond::Time_t> iovs;
  if( hasOptionValue( "hashes" ) ){
    std::string hs = getOptionValue<std::string>("hashes");
    tokenizer tok(hs);
    for( auto &t:tok ){
      hashes.push_back(t);
    }
  } else if ( hasOptionValue("tag") ){
    tag = getOptionValue<std::string>("tag");
    if(!hasOptionValue("iovs")) {
      std::cout <<"ERROR: no iovs provided for tag "<<tag<<std::endl;
      return 1;
    }
    std::string siovs = getOptionValue<std::string>("iovs");
    tokenizer tok( siovs );
    for( auto &t:tok ){
      iovs.push_back( boost::lexical_cast<unsigned long long>(t) );
    }
  }
   
  if(hashes.empty() and iovs.empty() ){
    std::cout <<"ERROR: no hashes or tag/iovs provided."<<std::endl;
    return 1;
  }

  persistency::ConnectionPool connPool;
  if( hasOptionValue("authPath") ){
    connPool.setAuthenticationPath( getOptionValue<std::string>( "authPath") ); 
  }
  connPool.configure();

  std::cout <<"# Connecting to source database on "<<connect<<std::endl;
  persistency::Session session;
  bool runTransaction = hasOptionValue("run");
  session = connPool.createSession( connect, false );
  session.transaction().start( true );
  if(hashes.empty()){
    cond::persistency::IOVProxy iovp;
    if(!runTransaction){
      iovp = session.readIov( tag );
    }
    for( auto &i: iovs ){
      persistency::Session iovSession = session;
      if( runTransaction ){
	std::cout <<"INFO: using run-labeled transactions."<<std::endl;
	iovSession = connPool.createReadOnlySession( connect, boost::lexical_cast<std::string>(i) );
        iovSession.transaction().start( true );
      }
      iovp = iovSession.readIov( tag );
      auto iov = iovp.getInterval( i );
      hashes.push_back( iov.payloadId );
      std::cout <<"INFO: found payload for iov "<<i<<" in tag "<<tag<<std::endl;
      if( runTransaction) iovSession.transaction().commit();
    }
  }

  cond::Binary data;
  cond::Binary info;
  std::string typeName("");
  for( auto& h:hashes ){
    bool found = session.fetchPayloadData( h, typeName, data, info );
    if( !found ) {
      std::cout <<"ERROR: payload for hash "<<h<<" has not been found."<<std::endl;
      return 2;
    } else {
      std::cout<<"INFO: retrieved payload for hash "<<h<<std::endl;
    }
  }
  session.transaction().commit();
  return 0;
}

int main( int argc, char** argv ){

  cond::TestReadUtilities utilities;
  return utilities.run(argc,argv);
}

