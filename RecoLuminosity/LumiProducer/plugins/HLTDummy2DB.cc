#ifndef RecoLuminosity_LumiProducer_HLTDummy2DB_h 
#define RecoLuminosity_LumiProducer_HLTDummy2DB_h 
#include "RecoLuminosity/LumiProducer/interface/DataPipe.h"
namespace lumi{
  class HLTDummy2DB : public DataPipe{
  public:
    HLTDummy2DB( const std::string& dest);
    virtual void retrieveRun( unsigned int );
    virtual const std::string dataType() const;
    virtual const std::string sourceType() const;
    virtual ~HLTDummy2DB();
  };//cl HLTDummy2DB
  //
  //implementation
  //
  HLTDummy2DB::HLTDummy2DB(const std::string& dest):DataPipe(dest){}
  void HLTDummy2DB::retrieveRun( unsigned int ){
  }
  const std::string HLTDummy2DB::dataType() const{
    return "HLT";
  }
  const std::string HLTDummy2DB::sourceType() const{
    return "DUMMY";
  }
  HLTDummy2DB::~HLTDummy2DB(){}
}//ns lumi
#include "RecoLuminosity/LumiProducer/interface/DataPipeFactory.h"
DEFINE_EDM_PLUGIN(lumi::DataPipeFactory,lumi::HLTDummy2DB,"HLTDummy2DB");
#endif
