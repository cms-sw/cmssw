#ifndef RecoLuminosity_LumiProducer_HLT2DB_h 
#define RecoLuminosity_LumiProducer_HLT2DB_h 
#include "RecoLuminosity/LumiProducer/interface/DataPipe.h"
namespace lumi{
  class HLT2DB : public DataPipe{
  public:
    HLT2DB(const std::string& dest);
    virtual void retrieveRun( unsigned int );
    virtual const std::string dataType() const;
    virtual const std::string sourceType() const;
    virtual ~HLT2DB();
  };//cl HLT2DB
  //
  //implementation
  //
  HLT2DB::HLT2DB(const std::string& dest):DataPipe(dest){}
  void HLT2DB::retrieveRun( unsigned int ){
  }
  const std::string HLT2DB::dataType() const{
    return "HLT";
  }
  const std::string HLT2DB::sourceType() const{
    return "DB";
  }
  HLT2DB::~HLT2DB(){}
}//ns lumi
#include "RecoLuminosity/LumiProducer/interface/DataPipeFactory.h"
DEFINE_EDM_PLUGIN(lumi::DataPipeFactory,lumi::HLT2DB,"HLT2DB");
#endif
