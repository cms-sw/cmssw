#ifndef RecoLuminosity_LumiProducer_TRGDummy2DB_h 
#define RecoLuminosity_LumiProducer_TRGDummy2DB_h 
#include "RecoLuminosity/LumiProducer/interface/DataPipe.h"
namespace lumi{
  class TRGDummy2DB : public DataPipe{
  public:
    TRGDummy2DB(const std::string& dest);
    virtual void retrieveRun( unsigned int );
    virtual const std::string dataType() const;
    virtual const std::string sourceType() const;
    virtual ~TRGDummy2DB();
  };//cl TRGDummy2DB
  //
  //implementation
  //
  TRGDummy2DB::TRGDummy2DB(const std::string& dest):DataPipe(dest){}
  void TRGDummy2DB::retrieveRun( unsigned int ){
  }
  const std::string TRGDummy2DB::dataType() const{
    return "TRG";
  }
  const std::string TRGDummy2DB::sourceType() const{
    return "DUMMY";
  }
  TRGDummy2DB::~TRGDummy2DB(){}
}//ns lumi
#include "RecoLuminosity/LumiProducer/interface/DataPipeFactory.h"
DEFINE_EDM_PLUGIN(lumi::DataPipeFactory,lumi::TRGDummy2DB,"TRGDummy2DB");
#endif
