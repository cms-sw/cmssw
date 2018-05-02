#include "CondCore/Utilities/interface/Utilities.h"
#include "CondCore/CondDB/interface/ConnectionPool.h"
#include "CondFormats/RunInfo/interface/LHCInfo.h"

//#include <boost/program_options.hpp>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace cond {
  class Dump_LHCInfo : public Utilities {
  public:
    Dump_LHCInfo();
    ~Dump_LHCInfo() override;
    int execute() override;
  };
}

cond::Dump_LHCInfo::Dump_LHCInfo():Utilities("conddb_dump_LHCInfo"){
  addConnectOption("connect","c","source connection string (optional, default=connect)");
  addOption<std::string>("tag","t","the source tag");
  addOption<std::string>("hash","i","the hash (id) of the payload to dump");
  addOption<cond::Time_t>("since","s","since time of the iov");
}

cond::Dump_LHCInfo::~Dump_LHCInfo(){
}

namespace Dump_LHCInfo_impl {
  void dump( const LHCInfo& payload, std::ostream& out ){
    std::stringstream ss;
    payload.print( ss );
    out <<ss.str();
    out <<std::endl;
  }
}

int cond::Dump_LHCInfo::execute(){

  std::string connect = getOptionValue<std::string>("connect");
  std::string tag("");
  
  cond::persistency::ConnectionPool connection;
  connection.setMessageVerbosity( coral::Error );
  connection.configure();

  cond::Hash payloadHash("");
  cond::Time_t since=0;
  if(hasOptionValue("hash")) {
    payloadHash = getOptionValue<std::string>("hash");
  } else {
    if(hasOptionValue("tag")){
      tag = getOptionValue<std::string>("tag");
    } else {
      std::cout <<"Error: no tag provided to identify the payload."<<std::endl;
      return 1;
    }
    if(hasOptionValue("since")){
      since = getOptionValue<cond::Time_t>("since");
    } else {
      std::cout <<"Error: no IOV since provided to identify the payload."<<std::endl;
      return 1;
    }
  }

  cond::persistency::Session session = connection.createSession( connect, false );
  session.transaction().start(true);

  if( payloadHash.empty() ){
    cond::persistency::IOVProxy iovSeq = session.readIov( tag );
    auto it = iovSeq.find( since );
    if( it == iovSeq.end() ){
      std::cout <<"Could not find iov with since="<<since<<" in tag "<<tag<<std::endl;
      session.transaction().commit();
      return 2;
    }
    payloadHash = (*it).payloadId;
  }

  std::shared_ptr<LHCInfo> payload = session.fetchPayload<LHCInfo>( payloadHash );
  session.transaction().commit();

  std::cout <<"# *********************************************************** "<<std::endl;
  std::cout <<"# Dumping payload id "<<payloadHash<<std::endl;
  std::cout <<"# *********************************************************** "<<std::endl;

  Dump_LHCInfo_impl::dump( *payload,std::cout );

  std::cout <<"# *********************************************************** "<<std::endl;
  return 0;
}

int main( int argc, char** argv ){

  cond::Dump_LHCInfo utilities;
  return utilities.run(argc,argv);
}
