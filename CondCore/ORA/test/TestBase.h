#ifndef INCLUDE_ORA_TESTBASE_H
#define INCLUDE_ORA_TESTBASE_H

#include "CondCore/ORA/interface/SchemaUtils.h"
#include "CondCore/ORA/interface/Exception.h"
#include <cstdlib>
//

namespace ora {

  class TestBase {
  public:
   
    explicit TestBase( const std::string& testName ):
      m_testName( testName ){}
      
    virtual ~TestBase(){
    }    

    virtual void execute( const std::string& connectionString ) = 0;

    void run( const std::string& connectionString ){
      std::string authpath("/afs/cern.ch/cms/DB/conddb/int9r");
      std::string pathenv(std::string("CORAL_AUTH_PATH=")+authpath);
      ::putenv(const_cast<char*>(pathenv.c_str()));
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
      }
    }

    void run(){
      std::string connStr("oracle://cms_orcoff_int/CMS_COND_UNIT_TESTS");
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


