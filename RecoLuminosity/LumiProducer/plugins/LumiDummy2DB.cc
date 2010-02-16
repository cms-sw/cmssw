#ifndef RecoLuminosity_LumiProducer_LumiDummy2DB_h 
#define RecoLuminosity_LumiProducer_LumiDummy2DB_h 
#include "RecoLuminosity/LumiProducer/interface/DataPipe.h"
namespace lumi{
  class LumiDummy2DB : public DataPipe{
  public:
    LumiDummy2DB(const std::string& dest);
    virtual void retrieveRun( unsigned int );
    virtual const std::string dataType() const;
    virtual const std::string sourceType() const;
    virtual ~LumiDummy2DB();
  };//cl LumiDummy2DB
  //
  //implementation
  //
  LumiDummy2DB::LumiDummy2DB( const std::string& dest):DataPipe(dest){}
  void LumiDummy2DB::retrieveRun( unsigned int ){
  }
  const std::string LumiDummy2DB::dataType() const{
    return "LUMI";
  }
  const std::string LumiDummy2DB::sourceType() const{
    return "DUMMY";
  }
  LumiDummy2DB::~LumiDummy2DB(){}
}//ns lumi
#include "RecoLuminosity/LumiProducer/interface/DataPipeFactory.h"
DEFINE_EDM_PLUGIN(lumi::DataPipeFactory,lumi::LumiDummy2DB,"LumiDummy2DB");
#endif
