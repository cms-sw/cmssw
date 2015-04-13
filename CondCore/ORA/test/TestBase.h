#ifndef INCLUDE_ORA_TESTBASE_H
#define INCLUDE_ORA_TESTBASE_H

#include "CondCore/ORA/interface/SchemaUtils.h"
#include "CondCore/ORA/interface/Exception.h"
#include <cstdlib>
#include <iostream>
//

namespace ora {

  class TestBase {
  public:

    static void sleep(){
      ::sleep(2);
    }

  public:
   
    explicit TestBase( const std::string& testName ):
      m_testName( testName ){}
      
    virtual ~TestBase(){
    }    

    virtual void execute( const std::string& connectionString ) = 0;

    void run( const std::string& connectionString ){
      const char* authEnv = ::getenv( "CORAL_AUTH_PATH" );
      std::string defaultPath("/afs/cern.ch/cms/DB/conddb");
      std::string pathEnv(std::string("CORAL_AUTH_PATH=")+defaultPath);
      if( !authEnv ){
	//setting environment variable: if pathEnv is defined in this scope (as it should be), it does not work!! (??)
	::putenv(const_cast<char*>(pathEnv.c_str()));
      }
      try{
	Serializer serializer;
	serializer.lock( connectionString );
	std::set<std::string> exclude;
	exclude.insert( Serializer::tableName());
	SchemaUtils::cleanUp( connectionString, exclude );
	execute( connectionString );
	serializer.release();
      }catch ( const std::exception& exc ){
	std::cout << "### TEST "<<m_testName<<" ERROR: "<<exc.what()<<std::endl;      
        exit(-1);
      }
    }

    void run(){
      std::string connStr("oracle://cms_orcoff_prep/CMS_COND_UNIT_TESTS");
      const char* envVar = ::getenv( "ORA_TEST_DB" );
      if( envVar ){
        connStr = std::string(envVar);
      }
      run( connStr );
    }

  protected:
    std::string m_testName;
			     
  };
  
}

#endif


