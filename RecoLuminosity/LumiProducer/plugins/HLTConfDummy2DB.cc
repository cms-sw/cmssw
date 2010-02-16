#ifndef RecoLuminosity_LumiProducer_HLTConfDummy2DB_h 
#define RecoLuminosity_LumiProducer_HLTConfDummy2DB_h 
#include "RecoLuminosity/LumiProducer/interface/DataPipe.h"
namespace lumi{
  class HLTConfDummy2DB : public DataPipe{
  public:
    explicit HLTConfDummy2DB(const std::string& dest);
    virtual void retrieveRun( unsigned int );
    virtual const std::string dataType() const;
    virtual const std::string sourceType() const;
    virtual ~HLTConfDummy2DB();
  };//cl HLTConfDummy2DB
  //
  //implementation
  //
  HLTConfDummy2DB::HLTConfDummy2DB(const std::string& dest):DataPipe(dest){}
  void HLTConfDummy2DB::retrieveRun( unsigned int ){
  }
  const std::string HLTConfDummy2DB::dataType() const{
    return "HLTCONF";
  }
  const std::string HLTConfDummy2DB::sourceType() const{
    return "DUMMY";
  }
  HLTConfDummy2DB::~HLTConfDummy2DB(){}
}//ns lumi
#include "RecoLuminosity/LumiProducer/interface/DataPipeFactory.h"
DEFINE_EDM_PLUGIN(lumi::DataPipeFactory,lumi::HLTConfDummy2DB,"HLTConfDummy2DB");
#endif
