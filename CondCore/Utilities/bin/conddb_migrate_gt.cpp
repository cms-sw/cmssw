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

// for the xml dump
#include "TFile.h"
#include "Cintex/Cintex.h"
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

cond::MigrateGTUtilities::MigrateGTUtilities():Utilities("conddb_test_gt_import"){
  addConnectOption("sourceConnect","s","source connection string(required)");
  addConnectOption("destConnect","d","destionation connection string(required)");
  addAuthenticationOptions();
  addOption<std::string>("globaltag","g","global tag (required)");
  addOption<bool>("verbose","v","verbose print out (optional)");
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
  bool debug = hasDebug();
  std::string destConnect = getOptionValue<std::string>("destConnect" );
  std::string sourceConnect = getOptionValue<std::string>("sourceConnect");
  bool verbose = hasOptionValue("verbose");

  std::vector<std::tuple<std::string,std::string,std::string,std::string,std::string> > gtlist;
  if(! getGTList( gtag, gtlist ) ) throw std::runtime_error( std::string("GT ")+gtag+" has not been found." );

  ConnectionPool connPool;
  if( hasDebug() ) connPool.setMessageVerbosity( coral::Debug );
  Session session = connPool.createSession( destConnect, true );
  session.transaction().start( false );

  GTEditor newGT = session.createGlobalTag( gtag );

  newGT.setDescription( "GT "+gtag+" migrated from account "+sourceConnect );
  newGT.setRelease( "CMSSW_6_2_0" );
  newGT.setSnapshotTime( boost::posix_time::microsec_clock::universal_time() );

  std::cout <<"Processing "<<gtlist.size()<<" tags."<<std::endl;
  for(auto gtitem : gtlist ){

    std::string tag = std::get<0>( gtitem );
    std::string payloadTypeName = std::get<1>( gtitem );
    std::string recordName = std::get<2>( gtitem );
    std::string recordLabel = std::get<3>( gtitem );
    std::string connectionString = std::get<4>( gtitem );
    
    std::cout <<"--> Processing tag "<<tag<<" (objectType: "<<payloadTypeName<<") on account "<<connectionString<<std::endl;
    auto connectionData = persistency::parseConnectionString( connectionString );
    std::string account = std::get<2>( connectionData );
    if( std::get<1>( connectionData )=="FrontierArc" ) {
      size_t len = account.size()-5;
      account = account.substr(0,len);
    }
    std::string sourceConn = "oracle://cms_orcon_adg/"+account;
    std::string destTag("");
    if(!session.checkMigrationLog( sourceConn, tag, destTag )){
      throw std::runtime_error("Tag "+tag+" from ["+sourceConn+"] has not been migrated to the destination database."); 
    }
 
    newGT.insert( recordName, recordLabel, tag );
  }
  newGT.flush(); 

  session.transaction().commit();
  std::cout <<"Global Tag \""<<gtag<<"\" imported."<<std::endl;

  return 0;
}

int main( int argc, char** argv ){

  cond::MigrateGTUtilities utilities;
  return utilities.run(argc,argv);
}

