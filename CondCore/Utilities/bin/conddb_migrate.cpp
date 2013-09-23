#include "CondCore/DBCommon/interface/DbSession.h"
//#include "CondCore/ORA/interface/Object.h"
#include "CondCore/DBCommon/interface/DbScopedTransaction.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/Auth.h"

#include "CondCore/CondDB/interface/Session.h"
#include "CondCore/CondDB/interface/IOVEditor.h"
#include "CondCore/CondDB/interface/IOVProxy.h"
//#include "CondCore/CondDB/interface/Streamer.h"

#include "CondCore/MetaDataService/interface/MetaData.h"

#include "CondCore/IOVService/interface/IOVProxy.h"

#include "CondCore/Utilities/interface/Utilities.h"
#include "CondCore/Utilities/interface/CondDBImport.h"
#include <iostream>

// for the xml dump
#include "TFile.h"
#include "Cintex/Cintex.h"
#include <sstream>

namespace conddb {

  class MigrateUtilities : public cond::Utilities {
    public:
      MigrateUtilities();
      ~MigrateUtilities();
      int execute();
  };
}

conddb::MigrateUtilities::MigrateUtilities():Utilities("conddb_import_tag"){
  addConnectOption("sourceConnect","s","source connection string(required)");
  addConnectOption("destConnect","d","destionation connection string(required)");
  addAuthenticationOptions();
  addOption<std::string>("tag","t","migrate only the tag (optional)");
  addOption<std::string>("newTag","n","name for the destination tag (optional)");

  ROOT::Cintex::Cintex::Enable();
}

conddb::MigrateUtilities::~MigrateUtilities(){
}

int conddb::MigrateUtilities::execute(){

  std::string newTag("");
  std::string tag("");
  if( hasOptionValue("tag")) {
    tag = getOptionValue<std::string>("tag");
    if( hasOptionValue("newTag")) newTag = getOptionValue<std::string>("newTag");
  }
  bool debug = hasDebug();
  std::string destConnect = getOptionValue<std::string>("destConnect" );

  std::string sourceConnect = getOptionValue<std::string>("sourceConnect");
  std::tuple<std::string,std::string,std::string> connPars = parseConnectionString( sourceConnect );
  if( std::get<0>( connPars ) == "frontier" ) throwException("Cannot migrate data from FronTier cache.","MigrateUtilities::execute");

  std::cout <<"# Connecting to source database on "<<sourceConnect<<std::endl;
  cond::DbSession sourcedb = openDbSession( "sourceConnect", cond::Auth::COND_READER_ROLE, true );
     
  if(debug){
    std::cout << "tag " << tag << std::endl;
  }  
  
  sourcedb.transaction().start( true );
  cond::MetaData  metadata(sourcedb);
  
  std::vector<std::string> tagToProcess;
  if( !tag.empty() ){
    tagToProcess.push_back( tag );
  } else {
    metadata.listAllTags( tagToProcess );
  }

  new_impl::Session session;
  std::cout <<"# Opening session on destination database..."<<std::endl;
  session.open( destConnect );
    
  session.transaction().start( false );
  if( !session.existsDatabase() ) session.createDatabase();
  session.transaction().commit();

  std::cout <<"# "<<tagToProcess.size()<<" tag(s) to process."<<std::endl;
  std::cout <<std::endl;
  size_t nt = 0;
  size_t tid = 0;
  for( auto t : tagToProcess ){
    tid++;
    std::cout <<"--> Processing tag["<<tid<<"]: "<<t<<std::endl;
    session.transaction().start( false );

    std::string destTag("");
    if( session.checkMigrationLog( sourceConnect, t, destTag ) ){
      std::cout <<"    Tag already migrated." << std::endl;
      session.transaction().commit();
      continue;
    }
    destTag = t;
    if( !newTag.empty() ){
      destTag = newTag;
    } else {
      if( session.existsIov( destTag ) ){
	destTag = destTag+"["+std::get<1>( connPars )+"/"+std::get<2>( connPars )+"]";
	std::cout <<"    Tag "<<t<<" already existing, renamed to "<<destTag<<std::endl;
      }
    }
    if( session.existsIov( destTag ) ){
      session.transaction().commit();
      throwException("Tag \""+destTag+"\" already exists.","MigrateUtilities::execute");
    }

    std::cout <<"    Resolving source tag oid..."<<std::endl;
    std::string iovTok = metadata.getToken(t); 
    if(iovTok.empty())
      throw std::runtime_error(std::string("tag ")+t+std::string(" not found") );
    std::map<std::string,Hash> tokenToHash;
    size_t niovs = 0;
    std::set<Hash> pids;
    new_impl::IOVEditor editor;
    std::cout <<"    Loading source tag..."<<std::endl;
    try{
      cond::IOVProxy sourceIov(sourcedb, iovTok);
      int tt = (int) sourceIov.timetype();
      std::string payloadType = *(sourceIov.payloadClasses().begin());
      std::cout <<"    Importing tag. Size:"<<sourceIov.size()<<" timeType:"<<cond::timeTypeNames(tt)<<" payloadObjectType=\""<<payloadType<<"\""<<std::endl;
      editor = session.createIov( destTag, (conddb::TimeType)tt, payloadType );
      editor.setDescription( "Tag "+t+" migrated from "+sourceConnect  );
      for(  auto iov : sourceIov ){
	Time_t s = iov.since();
	std::string tok = iov.token();
	Hash pid("");
	auto f = tokenToHash.find( tok );
	if( f == tokenToHash.end() ){
	  if(hasDebug() ) std::cout <<"Debug: fetching object for iov ["<<niovs+1<<"] oid: \""<<t<<"\" from source database"<<std::endl;
	  ora::Object obj = sourcedb.getObject( tok );
	  if(hasDebug() ) std::cout <<"Debug: importing object into destination database"<<std::endl;
	  pid = import( obj.typeName(), obj.address(), session );  
	  tokenToHash.insert( std::make_pair( tok, pid ) );
	  obj.destruct();
	} else {
	  pid= f->second;
	}
	pids.insert( pid );
	editor.insert( s, pid );
	niovs++;
	if( niovs && (niovs%1000==0) ) std::cout <<"    Total of iov inserted: "<<niovs<<std::endl;
      } 
      std::cout <<"    Total of iov inserted: "<<niovs<<std::endl;
      std::cout <<"    Flushing changes..."<<std::endl;
      editor.flush();
      session.addToMigrationLog( sourceConnect, t, destTag );
      session.transaction().commit();
      std::cout <<"    Tag \""<<t<<"\" imported. Payloads:"<<pids.size()<<" IOVs:"<<niovs<<std::endl;
      nt++;
    } catch ( const std::exception& e ){
      std::cout <<"    ERROR:"<<e.what()<<std::endl;
      std::cout <<"    Tag "<<t<<" will be skipped."<<std::endl;
      session.transaction().commit();
      continue;
    }
  }

  std::cout <<"# "<<nt<<" tag(s) migrated."<<std::endl;

  sourcedb.transaction().commit();

  return 0;
}

int main( int argc, char** argv ){

  conddb::MigrateUtilities utilities;
  return utilities.run(argc,argv);
}

