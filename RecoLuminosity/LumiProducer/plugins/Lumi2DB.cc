#ifndef RecoLuminosity_LumiProducer_Lumi2DB_h 
#define RecoLuminosity_LumiProducer_Lumi2DB_h 
#include "RecoLuminosity/LumiProducer/interface/DataPipe.h"
namespace lumi{
  class Lumi2DB : public DataPipe{
  public:
    Lumi2DB(const std::string& dest);
    virtual void retrieveData( unsigned int );
    virtual const std::string dataType() const;
    virtual const std::string sourceType() const;
    virtual ~Lumi2DB();
  };//cl Lumi2DB
  //
  //implementation
  //
  Lumi2DB::Lumi2DB(const std::string& dest):DataPipe(dest){}
  void Lumi2DB::retrieveData( unsigned int ){
  }
  const std::string Lumi2DB::dataType() const{
    return "LUMI";
  }
  const std::string Lumi2DB::sourceType() const{
    return "DB";
  }
  Lumi2DB::~Lumi2DB(){}
}//ns lumi
#include "RecoLuminosity/LumiProducer/interface/DataPipeFactory.h"
DEFINE_EDM_PLUGIN(lumi::DataPipeFactory,lumi::Lumi2DB,"Lumi2DB");
#endif
