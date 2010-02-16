#ifndef RecoLuminosity_LumiProducer_TRG2DB_h 
#define RecoLuminosity_LumiProducer_TRG2DB_h 
#include "RecoLuminosity/LumiProducer/interface/DataPipe.h"
namespace lumi{
  class TRG2DB : public DataPipe{
  public:
    TRG2DB(const std::string& dest);
    virtual void retrieveRun( unsigned int );
    virtual const std::string dataType() const;
    virtual const std::string sourceType() const;
    virtual ~TRG2DB();
  };//cl TRG2DB
  //
  //implementation
  //
 TRG2DB::TRG2DB(const std::string& dest):DataPipe(dest){}
  void TRG2DB::retrieveRun( unsigned int ){
  }
  const std::string TRG2DB::dataType() const{
    return "TRG";
  }
  const std::string TRG2DB::sourceType() const{
    return "DB";
  }
  TRG2DB::~TRG2DB(){}
}//ns lumi
#include "RecoLuminosity/LumiProducer/interface/DataPipeFactory.h"
DEFINE_EDM_PLUGIN(lumi::DataPipeFactory,lumi::TRG2DB,"TRG2DB");
#endif
