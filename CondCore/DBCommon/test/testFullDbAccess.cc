#include "CondCore/DBCommon/interface/DbConnection.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "CondCore/DBCommon/interface/DbScopedTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/Auth.h"
#include "CondCore/DBCommon/test/PayObj.h"
#include "CondCore/ORA/test/TestBase.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"

#include <iostream>
#include <boost/filesystem.hpp>

std::string AUTHPATH("/afs/cern.ch/cms/DB/conddb/test/key");

namespace cond {

  std::string getAuthPath(){
    std::string ret = AUTHPATH;
    const char* authEnv = ::getenv( "COND_AUTH_PATH" );
    if( authEnv ) ret = std::string(authEnv);
    return ret;
  }

  class TestDbAccess : public ora::TestBase {
  public:
    TestDbAccess(): 
      TestBase( "testFullDbAccess" ){
      connection.configuration() = cond::DbConnectionConfiguration::defaultConfigurations()[cond::CmsDefaults];
      connection.configuration().setAuthenticationPath( getAuthPath() );
      // has to be removed from 6.0 onwards...
      connection.configuration().setAuthenticationSystem( CondDbKey );
      connection.configuration().setMessageLevel( coral::Debug );
      connection.configure();
    }

    virtual ~TestDbAccess(){
    }

    int execute( const std::string& connstr ){
      if ( !boost::filesystem::exists( getAuthPath() ) ) {
	// sending a WARNING (and not an ERROR!) when the authentication path is not in the 'environment': in this case this test cannot be executed...
	std::cout<<"WARNING: Can't run this test, since the expected path \""<<getAuthPath()<<"\n has not been found in the execution filesystem. "<<std::endl;
	return 0;
      }
      
      unsigned int nw = 20;
      std::vector<std::pair<std::string,int> > wrIds;
      wrIds.reserve(nw);
      std::cout <<"# writing "<<nw<<" object into the DB..."<<std::endl; 
      {
	cond::DbSession session = connection.createSession();
	session.open( connstr, Auth::COND_WRITER_ROLE );
	cond::DbScopedTransaction trans( session );
	trans.start();

	session.createDatabase();
	std::vector<PayObj*> buff;
	std::string contName("PayObj_test");
	for(unsigned int i=0;i<nw;i++){
	  unsigned int gid = i+100;
	  PayObj* obj = new PayObj(gid);
	  buff.push_back(obj);
	  wrIds.push_back(std::make_pair(session.storeObject(obj,contName), gid));
	}
	session.flush();
	for(size_t i=0;i<buff.size();i++){
	  delete buff[i];
	}
	trans.commit();
      }
      sleep();
      std::cout <<"# reading back: trying with the default role..."<<std::endl; 
      {
	cond::DbSession session = connection.createSession();
	session.open( connstr, true  );
	session.transaction().start( true );

	std::string contClass = session.classNameForItem( wrIds[0].first );
	if( contClass != "PayObj"){
	  std::cout <<"ERROR: the class name found for the item stored is: "<<contClass<<std::endl;
	  return 1;
	}
	std::cout <<"# Class name for OId="<<wrIds[0].first<<" is:\""<<contClass<<"\"."<<std::endl;
	session.transaction().commit();
      } 
      sleep();
      std::cout <<"# reading back: going for the read only..."<<std::endl; 
      {
	cond::DbSession session = connection.createSession();
	session.open( connstr, true );
	session.transaction().start( true );

	for(size_t i=0;i<wrIds.size();i++){
	  std::string oid = wrIds[i].first;
	  unsigned int gid = wrIds[i].second;
	  boost::shared_ptr<PayObj>  obj = session.getTypedObject<PayObj>( oid );
	  PayObj ref(gid);
	  if( ref!=*obj ){
	    std::cout <<"ERROR: object read different from expected. OId="<<oid<<std::endl;
	    return 1;	    
	  }
	}
	std::cout <<"# Read back completed without errors."<<std::endl;
	session.transaction().commit();
      }  
      return 0;   
  
    }
    // shared connection
    DbConnection connection;
  };
}

int main( int argc, char** argv){
  edmplugin::PluginManager::Config config;
  edmplugin::PluginManager::configure(edmplugin::standard::config());
  cond::TestDbAccess test;
  return test.run();
}

