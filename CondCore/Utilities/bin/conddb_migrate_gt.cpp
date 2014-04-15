#include "CondCore/DBCommon/interface/DbSession.h"
#include "CondCore/DBCommon/interface/DbScopedTransaction.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/Auth.h"

#include "CondCore/CondDB/interface/ConnectionPool.h"
#include "CondCore/CondDB/interface/Session.h"
#include "CondCore/CondDB/interface/Utils.h"

#include "CondCore/CondDB/src/DbCore.h"

#include "CondCore/MetaDataService/interface/MetaData.h"

#include "CondCore/IOVService/interface/IOVProxy.h"

#include "CondCore/Utilities/interface/Utilities.h"
#include "CondCore/Utilities/interface/CondDBImport.h"
#include <iostream>

//
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/ITableDataEditor.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ICursor.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/AttributeList.h"

#include <sstream>
#include <vector>
#include <tuple>

// table definition
#define d_table( NAME ) namespace NAME {\
    static std::string tname;\
    }\
    namespace NAME

#define d_column( NAME, TYPE ) struct NAME {\
    static constexpr char const* name = #NAME;	\
    typedef TYPE type;\
    static std::string tableName(){ return tname; }\
    static std::string fullyQualifiedName(){ return tname+"."+name; }\
    };\


namespace cond {

  d_table( GT ){
    d_column( tagid, int );
  }

  d_table( TAGINV ){
    d_column( tagname, std::string );
    d_column( objectname, std::string );
    d_column( recordname, std::string );
    d_column( labelname, std::string );
    d_column( pfn, std::string );
    d_column( tagid, int );
  }

  class MigrateGTUtilities : public cond::Utilities {
    public:
      MigrateGTUtilities();
      bool getGTList( const std::string& gt, std::vector<std::tuple<std::string,std::string,std::string,std::string,std::string> >&tagList );
      int execute();
  };

  using namespace persistency;
}

cond::MigrateGTUtilities::MigrateGTUtilities():Utilities("conddb_migrate_gt"){
  addConnectOption("sourceConnect","s","source connection string (required)");
  addConnectOption("destConnect","d","destionation connection string (required)");
  addAuthenticationOptions();
  addOption<std::string>("globaltag","g","global tag (required)");
  addOption<std::string>("release","r","release validity (required)");
  addOption<bool>("verbose","v","verbose print out (optional)");
  addOption<bool>("dryRun","n","only display the actions (optional)");
}

bool cond::MigrateGTUtilities::getGTList( const std::string& gt, 
					   std::vector<std::tuple<std::string,std::string,std::string,std::string,std::string> >&tagList ){
  cond::DbSession gtSession =  openDbSession("sourceConnect",cond::Auth::COND_READER_ROLE,true);
  gtSession.transaction().start(true);
  coral::ISchema& schema = gtSession.nominalSchema();

  std::string gtTable("TAGTREE_TABLE_");
  gtTable += gt;
  if( !schema.existsTable( gtTable ) ){
    std::cout <<"ERROR: The specified Global Tag \"" << gt <<"\" has not been found in the database." <<std::endl;
    return false;
  }

  bool ret = false;
  GT::tname = gtTable;
  TAGINV::tname = "TAGINVENTORY_TABLE";

  persistency::Query< TAGINV::tagname, TAGINV::objectname, TAGINV::recordname, TAGINV::labelname, TAGINV::pfn > q( schema );
  q.addCondition<GT::tagid, TAGINV::tagid>();
  q.addOrderClause<TAGINV::tagname>();
  for ( auto row : q ) {
    tagList.push_back( row );
    ret = true;
  }
  
  gtSession.transaction().commit();
  return ret;
}

int cond::MigrateGTUtilities::execute(){

  std::string gtag = getOptionValue<std::string>("globaltag");
  std::string release = getOptionValue<std::string>("release");
  bool debug = hasDebug();
  std::string destConnect = getOptionValue<std::string>("destConnect" );
  std::string sourceConnect = getOptionValue<std::string>("sourceConnect");
  bool verbose = hasOptionValue("verbose");
  bool dryRun = hasOptionValue("dryRun");

  std::vector<std::tuple<std::string,std::string,std::string,std::string,std::string> > gtlist;
  if(! getGTList( gtag, gtlist ) ) throw std::runtime_error( std::string("Source GT ")+gtag+" has not been found." );

  ConnectionPool connPool;
  if( hasDebug() ) connPool.setMessageVerbosity( coral::Debug );
  Session session = connPool.createSession( destConnect, !dryRun );
  session.transaction().start( dryRun );

  if( session.existsGlobalTag( gtag ) ){
    std::cout <<"GT "<<gtag<<" already exists in the destination database."<<std::endl;
    return 1;
  }
  GTEditor newGT;
  if( !dryRun ){
    newGT = session.createGlobalTag( gtag );
    newGT.setDescription( "GT "+gtag+" migrated from account "+sourceConnect );
    newGT.setRelease( release );
    newGT.setSnapshotTime( boost::posix_time::microsec_clock::universal_time() );
  }
  std::cout <<"Processing "<<gtlist.size()<<" tags."<<std::endl;
  size_t nerr = 0;
  for(auto gtitem : gtlist ){

    std::string tag = std::get<0>( gtitem );
    std::string payloadTypeName = std::get<1>( gtitem );
    std::string recordName = std::get<2>( gtitem );
    std::string recordLabel = std::get<3>( gtitem );
    std::string connectionString = std::get<4>( gtitem );
    
    std::cout <<"--> Processing tag "<<tag<<" (objectType: "<<payloadTypeName<<") on account "<<connectionString<<std::endl;

    std::string sourceConn = connectionString;
    std::string protocol = getConnectionProtocol( connectionString );
    if( protocol == "frontier" ){
      sourceConn = convertoToOracleConnection( connectionString);
    }

    std::string destTag("");
    cond::MigrationStatus status;
    bool exists = session.checkMigrationLog( sourceConn, tag, destTag, status );
    if(!exists || status==cond::ERROR){
      std::cout <<"    ERROR: Tag "<<tag<<" from "<<sourceConn<<" has not been migrated to the destination database."<<std::endl; 
      if( !dryRun ){
	return 1;
      } else {
	nerr++;
      }
    } else {
      std::cout <<"    Inserting tag "<<destTag<<std::endl; 
    }
    if( !dryRun ) newGT.insert( recordName, recordLabel, tag );
  }
  if( !dryRun )newGT.flush(); 

  session.transaction().commit();
  std::cout << std::endl;
  if( !dryRun ) {
    std::cout <<"Global Tag \""<<gtag<<"\" imported."<<std::endl;
  } else {
    std::cout <<"Importing Global Tag \""<<gtag<<"\" will run with "<<nerr<<" error(s)"<<std::endl; 
  }
  return 0;
}

int main( int argc, char** argv ){

  cond::MigrateGTUtilities utilities;
  return utilities.run(argc,argv);
}

