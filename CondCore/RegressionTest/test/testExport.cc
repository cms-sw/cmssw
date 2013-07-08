#include "CondCore/Utilities/interface/ExportIOVUtilities.h"
#include "CondCore/RegressionTest/interface/TestFunct.h"

namespace cond_regression {

  class ExportIOVTest : public cond::ExportIOVUtilities {
  public:
    ExportIOVTest();
    ~ExportIOVTest();
    int execute();
  private:
    TestFunct m_tf;
  };
}

cond_regression::ExportIOVTest::ExportIOVTest():
  cond::ExportIOVUtilities("testExport"){
  addOption<std::string>("initDatabase","I","initialize the database with the specified tag");
  addOption<bool>("cleanUp","C","initialize cleanUp the database account");
  addOption<std::string>("read","R","read and verify the specified tag");
  addOption<int>("seed","Z","input seed for data generation");
  addOption<bool>("metadata","M","initialize the database with the metadata table");
  addOption<bool>("export","E","start the export with the specified parameters");
}

cond_regression::ExportIOVTest::~ExportIOVTest(){
}

int cond_regression::ExportIOVTest::execute(){
  // TestFunct returns false in case of success!
  if( hasOptionValue("initDatabase") ){
    m_tf.s = openDbSession("sourceConnect");
    std::string tag = getOptionValue<std::string>("initDatabase");
    int seed = -1;
    if(hasOptionValue("seed")){
      seed = getOptionValue<int>("seed");
    } else {
      long now = ::time(NULL);
      int low = now%100;
      ::srand( low );
      seed = ::rand()%100;
    }
    cond::Time_t since = 1;
    if( hasOptionValue("beginTime") ){
      since = getOptionValue<cond::Time_t>("beginTime");
    }
    if(!m_tf.DropTables( m_tf.s.connectionString() )){
      return 1;
    }
    bool withTestMetadata = false;
    if( hasOptionValue("metadata") ) {
      withTestMetadata = true;
      if(!m_tf.CreateMetaTable()){
	return 1;
      }
    }
    if(!m_tf.WriteWithIOV(tag, seed, since, withTestMetadata)){
      return 1;
    }
    return 0;
  }
  if( hasOptionValue("cleanUp") ){
    m_tf.s = openDbSession("sourceConnect");
    if(!m_tf.DropTables( m_tf.s.connectionString() )){
      return 1;
    }
    return 0;
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
      std::cout <<"## Object from tag="<<tag<<" seed="<<metadata.first<<" validity="<<metadata.second<<" from target database READ"<<(ret?" ":" NOT ")<<"OK "<<std::endl;
    }
    if(!ret){
      return 1;
    }
    return 0;
  }
  if( hasOptionValue("export") ){
    return cond::ExportIOVUtilities::execute();
  }
  return 1;
}

int main( int argc, char** argv ){
  cond_regression::ExportIOVTest test;
  return test.run(argc,argv);
}
