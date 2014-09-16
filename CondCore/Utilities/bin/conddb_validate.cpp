#include "CondCore/DBCommon/interface/DbSession.h"
#include "CondCore/DBCommon/interface/DbScopedTransaction.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/Auth.h"

#include "CondCore/CondDB/interface/ConnectionPool.h"
#include "CondCore/CondDB/interface/Utils.h"
#include "CondCore/CondDB/interface/IOVEditor.h"
#include "CondCore/CondDB/interface/IOVProxy.h"

#include "CondCore/MetaDataService/interface/MetaData.h"

#include "CondCore/IOVService/interface/IOVProxy.h"

#include "CondCore/Utilities/interface/Utilities.h"
#include "CondCore/Utilities/interface/CondDBImport.h"
#include <iostream>
#include <boost/filesystem.hpp>

// for the xml dump
#include "TFile.h"
#include "Cintex/Cintex.h"
#include <sstream>
#include <stdio.h>

namespace cond {

  class ValidateUtilities : public cond::Utilities {
    public:
      ValidateUtilities();
      ~ValidateUtilities();
      int execute();
  };

  std::string writeTag( const std::string& tag, const std::string& label, persistency::Session& session, std::string tmpDir ){
    session.transaction().start();
    persistency::IOVProxy p = session.readIov( tag, true );
    std::cout <<"    "<<p.loadedSize()<<" iovs loaded."<<std::endl;
    std::string destFile = label+".db";
    boost::filesystem::path destFilePath( destFile );
    if( !tmpDir.empty() ) {
      destFilePath = boost::filesystem::path( tmpDir ) / destFilePath;
    }
    destFile = destFilePath.string();
    persistency::ConnectionPool localPool;
    persistency::Session writeSession = localPool.createSession( "sqlite_file:"+destFile, true );
    writeSession.transaction().start( false );
    persistency::IOVEditor writeEditor = writeSession.createIov( p.payloadObjectType(), tag, p.timeType(), p.synchronizationType() );
    writeEditor.setValidationMode();
    writeEditor.setDescription("Validation");
    for( auto iov : p ){
      std::pair<std::string,boost::shared_ptr<void> > readBackPayload = fetch( iov.payloadId, session );
      cond::Hash ph = import( readBackPayload.first, readBackPayload.second.get(), writeSession );
      writeEditor.insert( iov.since, ph );
    }
    writeEditor.flush();
    writeSession.transaction().commit();
    session.transaction().commit();
    return destFile;
  }

  bool compareFiles( const std::string& refFileName, const std::string& candFileName ){
    FILE* refFile = fopen( refFileName.c_str(), "r" );
    if( refFile == NULL ){
      throwException("Can't open file "+refFileName, "compareFiles" ); 
    }
    FILE* candFile = fopen( candFileName.c_str(), "r" );
    if( candFile == NULL ){
      throwException("Can't open file "+candFileName, "compareFiles" ); 
    }
    int N = 10000;
    char buf1[N];
    char buf2[N];
    
    bool cmpOk = true;
    do {
      size_t r1 = fread( buf1, 1, N, refFile );
      size_t r2 = fread( buf2, 1, N, candFile );
      
      if( r1 != r2 || memcmp( buf1, buf2, r1)) {
	cmpOk = false;
	break;
      }
    } while(!feof(refFile) || !feof(candFile));
    
    fclose(refFile);
    fclose( candFile );

    return cmpOk;
  }

  void flushFile( const std::string& account, const std::string& tag, const std::string& fileName ){
    boost::filesystem::path accountDir( account );
    if( !boost::filesystem::exists( accountDir ) ) boost::filesystem::create_directory( accountDir );
    boost::filesystem::path tagDir( tag.c_str() );
    tagDir = accountDir / tagDir;
    if( !boost::filesystem::exists( tagDir ) ) boost::filesystem::create_directory( tagDir );
    boost::filesystem::path sourceFilePath( fileName );
    boost::filesystem::path destFilePath = tagDir / sourceFilePath.leaf();
    if( boost::filesystem::exists(destFilePath) ) boost::filesystem::remove(destFilePath);
    boost::filesystem::copy_file( sourceFilePath, destFilePath  );
  }

  void cleanUp( const std::string& fileName ){
    boost::filesystem::path fp( fileName );
    if( boost::filesystem::exists(fp) ) boost::filesystem::remove( fp );
  }

}

cond::ValidateUtilities::ValidateUtilities():Utilities("conddb_validate"){
  addConnectOption("reference","r","reference database connection string (required)");
  addConnectOption("candidate","c","candidate 1 connection string (required)");
  addAuthenticationOptions();
  addOption<std::string>("tag","t","migrate only the tag (optional)");
  addOption<std::string>("dir","d","tmp folder to dump the temporary files for the comparison (optional)");
  ROOT::Cintex::Cintex::Enable();
}

cond::ValidateUtilities::~ValidateUtilities(){
}

int cond::ValidateUtilities::execute(){

  initializeForDbConnection();
  std::string tag("");
  if( hasOptionValue("tag") ) tag = getOptionValue<std::string>("tag");
  std::string dir("");
  if( hasOptionValue("dir") ) dir = getOptionValue<std::string>("dir");

  bool debug = hasDebug();

  std::string refConnect = getOptionValue<std::string>("reference");
  std::string candidate = getOptionValue<std::string>("candidate" );

  std::tuple<std::string,std::string,std::string> connPars = persistency::parseConnectionString( refConnect );
  if( std::get<0>( connPars ) == "frontier" ) throwException("Cannot validate data from FronTier cache.","ValidateUtilities::execute");
     
  std::string refDbName = std::get<1>( connPars )+"_"+std::get<2>( connPars );
  
  if(debug){
    std::cout << "tag " << tag << std::endl;
  }  
  
  
  std::vector<std::string> tagToProcess;
  if( !tag.empty() ){
    tagToProcess.push_back( tag );
  } else {
    cond::DbSession refdb = openDbSession( "reference", cond::Auth::COND_READER_ROLE, true );
    refdb.transaction().start( true );
    cond::MetaData  metadata(refdb);
    metadata.listAllTags( tagToProcess );
    refdb.transaction().commit();
  }

  persistency::ConnectionPool connPool;
  std::cout <<"# Opening session on reference database..."<<std::endl;
  persistency::Session session0 = connPool.createSession( refConnect );
  std::cout <<"# Opening session on candidates database..."<<std::endl;
  persistency::Session session1 = connPool.createSession( candidate );
  session1.transaction().start();
  if( !session1.existsDatabase() ) throwException( "Candidate DB \""+candidate+" does not exist.",
						   "MigrateUtilities::execute" );

  std::cout <<"# "<<tagToProcess.size()<<" tag(s) to process."<<std::endl;
  std::cout <<std::endl;
  size_t nt = 0;
  size_t tid = 0;
  for( auto t : tagToProcess ){
    tid++;
    std::cout <<"--> Processing tag["<<tid<<"]: "<<t<<std::endl;
    std::string refFileName("");
    std::string candFileName("");
    try{      
      std::cout <<"    Writing reference db"<<std::endl;
      refFileName = writeTag( t, "ref", session0, dir );
      std::string destTag("");
      cond::MigrationStatus status;
      if( !session1.checkMigrationLog( refConnect, t, destTag, status ) ) {
	std::cout << "    ERROR: Tag "<< t <<" has not been migrated in database " << candidate <<std::endl;
	boost::filesystem::remove( boost::filesystem::path(refFileName) );
      } else {
	std::cout <<"    Writing candidate db"<<std::endl;
	candFileName = writeTag( destTag, "cand", session1, dir ); 
	bool cmp = compareFiles( refFileName, candFileName ); 
	if(!cmp){
	  std::cout <<"    ERROR: Comparison found differences."<<std::endl;
	  flushFile( refDbName, t, refFileName );
	  flushFile( refDbName, t, candFileName  );
	} else {
	  std::cout <<"    Comparison OK."<<std::endl;
	}
	nt++;
      }
    } catch ( const std::exception& e ){
      std::cout <<"    ERROR:"<<e.what()<<std::endl;
      std::cout <<"    Tag "<<t<<" will be skipped."<<std::endl;
    }
    cleanUp( refFileName );
    cleanUp( candFileName );
  }

  std::cout <<std::endl<<"# "<<nt<<" tag(s) checked for validation."<<std::endl;

  return 0;
}

int main( int argc, char** argv ){

  cond::ValidateUtilities utilities;
  return utilities.run(argc,argv);
}

