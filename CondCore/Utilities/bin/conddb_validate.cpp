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

  class ValidateUtilities : public cond::Utilities {
    public:
      ValidateUtilities();
      ~ValidateUtilities();
      int execute();
  };

}

cond::ValidateUtilities::ValidateUtilities():Utilities("conddb_validate2"){
  addConnectOption("sourceConnect","s","source connection string(required)");
  addConnectOption("destConnect","d","destionation connection string(required)");
  addAuthenticationOptions();
  addOption<std::string>("tag","t","migrate only the tag (optional)");
  addOption<std::string>("dir","d","tmp folder to dump the temporary files for the comparison (optional)");
}

cond::ValidateUtilities::~ValidateUtilities(){
}

int cond::ValidateUtilities::execute(){

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
  std::string destConnect = getOptionValue<std::string>("destConnect" );
  std::string sourceConnect = getOptionValue<std::string>("sourceConnect");

  std::tuple<std::string,std::string,std::string> connPars = persistency::parseConnectionString( sourceConnect );
  if( std::get<0>( connPars ) == "frontier" ) throwException("Cannot validate data from FronTier cache.","MigrateUtilities::execute");

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

  persistency::ConnectionPool connPool;
  if( hasDebug() ) {
    connPool.setMessageVerbosity( coral::Debug );
    connPool.configure();
  }
  persistency::Session sourceSession = connPool.createSession( sourceConnect );

  std::cout <<"# Opening session on destination database..."<<std::endl;
  persistency::Session destSession = connPool.createSession( destConnect );
   
  bool existsDestDb = false; 
  destSession.transaction().start( true );
  existsDestDb = destSession.existsDatabase();
  destSession.transaction().commit();
  if( !existsDestDb ) {
    std::cout <<"# ERROR: Database does not exist in the destination "<<destConnect<<std::endl;
    return 1;
  }

  std::cout <<"# "<<tagToProcess.size()<<" tag(s) to process."<<std::endl;
  std::cout <<std::endl;
  size_t nt = 0;
  size_t nt_validated = 0;
  size_t nt_invalid = 0;
  for( auto t : tagToProcess ){
    nt++;
    std::cout <<"--> Processing tag["<<nt<<"]: "<<t<<std::endl;

    std::string destTag("");
    cond::MigrationStatus status = ERROR;
    destSession.transaction().start( true );
    bool existsEntry = destSession.checkMigrationLog( sourceConnect, t, destTag, status );
    if( existsEntry ){
      std::cout <<"    Tag found. Current status="<<validationStatusText[status]<<std::endl;
    } else {
      std::cout <<"    Tag not migrated."<<std::endl;
      continue;
    }
    destSession.transaction().commit();
    if( existsEntry && status == ERROR ){
      std::cout <<"    Tag has been migrated with Errors."<<std::endl;
      continue;
    }
    //if( status == MIGRATED ){
    try{
      if( validateTag( t, sourceSession, destTag, destSession ) ){
	std::cout <<"    Tag validated."<<std::endl;
	nt_validated++;
      } else {
	std::cout <<"    ERROR: Migrated tag different from reference."<<std::endl;
	nt_invalid++;
      }
    } catch ( const std::exception& e ){
      std::cout <<"    ERROR in validation: "<<e.what()<<std::endl;
    }
    //}
    //try{
    //  persistency::TransactionScope usc( destSession.transaction() );
    //  usc.start( false );
    //  if( existsEntry ) {
    //	destSession.updateMigrationLog( sourceConnect, t, status );
    //  } else {
    //	destSession.addToMigrationLog( sourceConnect, t, destTag, status );   
    //  }
    //  usc.commit();
    //} catch ( const std::exception& e ){
    //  std::cout <<"    ERROR updating the status: "<<e.what()<<std::endl;
    //}
  }

  std::cout <<"# "<<nt<<" tag(s) processed. Validated: "<<nt_validated<<" Invalid:"<<nt_invalid<<std::endl;

  return 0;
}

int main( int argc, char** argv ){

  cond::ValidateUtilities utilities;
  return utilities.run(argc,argv);
}

