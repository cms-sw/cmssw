#include "CondCore/DBCommon/interface/DbSession.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
//#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/Auth.h"

#include "CondCore/CondDB/interface/ConnectionPool.h"
#include "CondCore/CondDB/interface/Utils.h"
#include "CondCore/CondDB/interface/IOVEditor.h"
#include "CondCore/CondDB/interface/IOVProxy.h"
#include "CondCore/CondDB/src/DbCore.h"

#include "CondCore/MetaDataService/interface/MetaData.h"

#include "CondCore/Utilities/interface/Utilities.h"
//#include "CondCore/Utilities/interface/CondDBImport.h"
#include "CondCore/Utilities/interface/CondDBTools.h"
#include <iostream>

// for the xml dump
#include "TFile.h"
#include <sstream>

namespace cond {

  class MigrateUtilities : public cond::Utilities {
    public:
      MigrateUtilities();
      ~MigrateUtilities();
      int execute();
  };

}

cond::MigrateUtilities::MigrateUtilities():Utilities("conddb_migrate"){
  addConnectOption("sourceConnect","s","source connection string(required)");
  addConnectOption("destConnect","d","destionation connection string(required)");
  addAuthenticationOptions();
  addOption<std::string>("tag","t","migrate only the tag (optional)");
  addOption<std::string>("newTag","n","name for the destination tag (optional)");
}

cond::MigrateUtilities::~MigrateUtilities(){
}

int cond::MigrateUtilities::execute(){

  bool debug = hasDebug();
  int filterPosition = -1;
  std::string tag("");
  if( hasOptionValue("tag")) {
    tag = getOptionValue<std::string>("tag");
    if(debug){
      std::cout << "tag " << tag << std::endl;
    }  
    if( tag[0] == '*' ){ 
      tag = tag.substr(1);
      filterPosition = 0;
    }
    if( tag[tag.size()-1] == '*' ){
      tag = tag.substr(0,tag.size()-1);
      filterPosition = 1;
    }
  }
  bool replace = hasOptionValue("replace");
  bool validate = !hasOptionValue("fast");

  std::string destConnect = getOptionValue<std::string>("destConnect" );
  std::string sourceConnect = getOptionValue<std::string>("sourceConnect");

  std::tuple<std::string,std::string,std::string> connPars = persistency::parseConnectionString( sourceConnect );
  if( std::get<0>( connPars ) == "frontier" ) throwException("Cannot migrate data from FronTier cache.","MigrateUtilities::execute");

  std::cout <<"# Connecting to source database on "<<sourceConnect<<std::endl;
  cond::DbSession sourcedb = openDbSession( "sourceConnect", cond::Auth::COND_READER_ROLE, true );
  sourcedb.transaction().start( true );
  cond::MetaData  metadata(sourcedb);
  std::vector<std::string> tagToProcess;
  if( !tag.empty() && filterPosition == -1 ){
    tagToProcess.push_back( tag );
  } else {
    metadata.listAllTags( tagToProcess );
    if( filterPosition != -1 ) {
      std::vector<std::string> filteredList;
      for( const auto& t: tagToProcess ) {
	size_t ptr = t.find( tag );
	if( ptr != std::string::npos && ptr < filterPosition ) filteredList.push_back( t );
      }
      tagToProcess = filteredList;
    }
  }

  cond::DbSession logdb = openDbSession("log", cond::Auth::COND_READER_ROLE, true ); 

  persistency::ConnectionPool connPool;
  if( hasDebug() ) {
    connPool.setMessageVerbosity( coral::Debug );
    connPool.configure();
  }
  persistency::Session sourceSession = connPool.createSession( sourceConnect );

  std::cout <<"# Opening session on destination database..."<<std::endl;
  persistency::Session destSession = connPool.createSession( destConnect, true, COND_DB );
    
  destSession.transaction().start( false );
  if( !destSession.existsDatabase() ) destSession.createDatabase();
  destSession.transaction().commit();

  std::cout <<"# "<<tagToProcess.size()<<" tag(s) to process."<<std::endl;
  std::cout <<std::endl;
  size_t nt = 0;
  size_t nt_migrated = 0;
  size_t nt_validated = 0;
  size_t nt_error = 0;
  for( auto t : tagToProcess ){
    nt++;
    std::cout <<"--> Processing tag["<<nt<<"]: "<<t<<std::endl;

    std::string destTag("");
    cond::MigrationStatus status = ERROR;
    destSession.transaction().start( false );
    bool existsEntry = destSession.checkMigrationLog( sourceConnect, t, destTag, status );
    if( existsEntry ){
      std::cout <<"    Tag already processed. Current status="<<validationStatusText[status]<<std::endl;
    } else {
      destTag = t;
      if( destSession.existsIov( destTag ) ){
	destTag = destTag+"["+std::get<1>( connPars )+"/"+std::get<2>( connPars )+"]";
	std::cout <<"    Tag "<<t<<" already existing, renamed to "<<destTag<<std::endl;	
      }
    }
    destSession.transaction().commit();
    if( !existsEntry || status == ERROR || replace ){
      try{
        persistency::UpdatePolicy policy = persistency::NEW;
	if( replace ) policy = persistency::REPLACE;
	migrateTag( t, sourceSession, destTag, destSession, policy, logdb );
	status = MIGRATED;
        nt_migrated++;
      } catch ( const std::exception& e ){
	nt_error++;
	std::cout <<"    ERROR in migration: "<<e.what()<<std::endl;
	std::cout <<"    Tag "<<t<<" has not been migrated."<<std::endl;
      }
    } 
    if( validate && status == MIGRATED ){
      try{
	if( validateTag( t, sourceSession, destTag, destSession ) ){
	  std::cout <<"    Tag validated."<<std::endl;
	  status = VALIDATED;
	  nt_validated++;
	} else {
	  std::cout <<"    ERROR: Migrated tag different from reference."<<std::endl;
	}
      } catch ( const std::exception& e ){
	std::cout <<"    ERROR in validation: "<<e.what()<<std::endl;
      }
    }
    try{
      persistency::TransactionScope usc( destSession.transaction() );
      usc.start( false );
      if( existsEntry ) {
	destSession.updateMigrationLog( sourceConnect, t, status );
      } else {
	destSession.addToMigrationLog( sourceConnect, t, destTag, status );   
      }
      usc.commit();
    } catch ( const std::exception& e ){
      std::cout <<"    ERROR updating the status: "<<e.what()<<std::endl;
    }
  }

  std::cout <<"# "<<nt<<" tag(s) processed. Migrated: "<<nt_migrated<<" Validated: "<<nt_validated<<" Errors:"<<nt_error<<std::endl;

  return 0;
}

int main( int argc, char** argv ){

  cond::MigrateUtilities utilities;
  return utilities.run(argc,argv);
}

