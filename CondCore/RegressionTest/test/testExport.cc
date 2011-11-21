#include "CondCore/Utilities/interface/Utilities.h"
#include "CondCore/RegressionTest/interface/TestFunct.h"

namespace cond_regression {

  class ExportIOVTest : public cond::Utilities {
  public:
    ExportIOVTest();
    ~ExportIOVTest();
    int execute();
  private:
    TestFunct m_tf;
  };
}

cond_regression::ExportIOVTest::ExportIOVTest():
  cond::Utilities("testExport"){
  addAuthenticationOptions();
  addConnectOption("sourceConnect","s","source connection string");
  addConnectOption("destConnect","d","destionation connection string");
  addOption<cond::Time_t>("beginTime","b","begin time (first since)");
  addOption<std::string>("initDatabase","I","initialize the database with the specified tag");
  addOption<bool>("cleanUp","C","initialize cleanUp the database account");
  addOption<std::string>("read","R","read and verify the specified tag");
  addOption<int>("seed","Z","input seed for data generation");
  addOption<bool>("metadata","M","initialize the database with the metadata table");
}

cond_regression::ExportIOVTest::~ExportIOVTest(){
}

int cond_regression::ExportIOVTest::execute(){
  // TestFunct returns false in case of success!
  if( hasOptionValue("initDatabase") ){
    if(!hasOptionValue("seed")){
      throw std::runtime_error("seed parameter has not been provided.");
    }
    if(!hasOptionValue("beginTime")){
      throw std::runtime_error("beginTime parameter has not been provided.");
    }
    m_tf.s = openDbSession("sourceConnect");
    std::string tag = getOptionValue<std::string>("initDatabase");
    int seed = getOptionValue<int>("seed");
    cond::Time_t since = getOptionValue<cond::Time_t>("beginTime");
    if(!m_tf.DropTables( m_tf.s.connectionString() )){
      if( hasOptionValue("metadata") ) m_tf.CreateMetaTable();
      return m_tf.WriteWithIOV(tag, seed, since);
    }
    return 1;
  }
  if( hasOptionValue("cleanUp") ){
    m_tf.s = openDbSession("sourceConnect");
    return m_tf.DropTables( m_tf.s.connectionString() );
  }
  if( hasOptionValue("read") ){
    std::string tag = getOptionValue<std::string>("read");
    m_tf.s = openDbSession("sourceConnect");
    std::pair<int,int> metadata = m_tf.GetMetadata( tag );
    if( hasOptionValue("destConnect") ){
      m_tf.s = openDbSession("destConnect");
    }
    bool ret = m_tf.ReadWithIOV( tag, metadata.first, metadata.second );
    if( hasDebug() ){
      std::cout <<"## Object from tag="<<tag<<" seed="<<metadata.first<<" validity="<<metadata.second<<" from target database READ"<<(ret?" NOT ":" ")<<"OK "<<std::endl;
    }
    return ret;
  }
  return 1;
}

int main( int argc, char** argv ){
  cond_regression::ExportIOVTest test;
  return test.run(argc,argv);
}
