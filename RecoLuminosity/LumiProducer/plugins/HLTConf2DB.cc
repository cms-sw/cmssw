#ifndef RecoLuminosity_LumiProducer_HLTConf2DB_h 
#define RecoLuminosity_LumiProducer_HLTConf2DB_h 
#include "RecoLuminosity/LumiProducer/interface/DataPipe.h"
namespace lumi{
  class HLTConf2DB : public DataPipe{
  public:
    explicit HLTConf2DB( const std::string& dest);
    virtual void retrieveData( unsigned int );
    virtual const std::string dataType() const;
    virtual const std::string sourceType() const;
    virtual ~HLTConf2DB();
  };//cl HLTConf2DB
  //
  //implementation
  //
  HLTConf2DB::HLTConf2DB( const std::string& dest):DataPipe(dest){}
  void HLTConf2DB::retrieveData( unsigned int ){
  }
  const std::string HLTConf2DB::dataType() const{
    return "HLTConf";
  }
  const std::string HLTConf2DB::sourceType() const{
    return "DB";
  }
  HLTConf2DB::~HLTConf2DB(){}
}//ns lumi
#include "RecoLuminosity/LumiProducer/interface/DataPipeFactory.h"
DEFINE_EDM_PLUGIN(lumi::DataPipeFactory,lumi::HLTConf2DB,"HLTConf2DB");
#endif
