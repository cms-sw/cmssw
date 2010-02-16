#ifndef RecoLuminosity_LumiProducer_RunSummary2DB_h 
#define RecoLuminosity_LumiProducer_RunSummary2DB_h 
#include "RecoLuminosity/LumiProducer/interface/DataPipe.h"
namespace lumi{
  class RunSummary2DB : public DataPipe{
  public:
    RunSummary2DB( const std::string& dest);
    virtual void retrieveRun( unsigned int );
    virtual const std::string dataType() const;
    virtual const std::string sourceType() const;
    virtual ~RunSummary2DB();

  };//cl RunSummary2DB
  //
  //implementation
  //
  RunSummary2DB::RunSummary2DB(const std::string& dest):DataPipe(dest){}
  void RunSummary2DB::retrieveRun( unsigned int ){
  }
  const std::string RunSummary2DB::dataType() const{
    return "RUNSUMMARY";
  }
  const std::string RunSummary2DB::sourceType() const{
    return "DB";
  }
  RunSummary2DB::~RunSummary2DB(){}
}//ns lumi
#include "RecoLuminosity/LumiProducer/interface/DataPipeFactory.h"
DEFINE_EDM_PLUGIN(lumi::DataPipeFactory,lumi::RunSummary2DB,"RunSummary2DB");
#endif
