#ifndef RecoLuminosity_LumiProducer_RunSummaryDummy2DB_h 
#define RecoLuminosity_LumiProducer_RunSummaryDummy2DB_h 
#include "RecoLuminosity/LumiProducer/interface/DataPipe.h"
namespace lumi{
  class RunSummaryDummy2DB : public DataPipe{
  public:
    RunSummaryDummy2DB(const std::string& dest);
    virtual void retrieveRun( unsigned int );
    virtual const std::string dataType() const;
    virtual const std::string sourceType() const;
    virtual ~RunSummaryDummy2DB();
 
  };//cl RunSummaryDummy2DB
  //
  //implementation
  //
  RunSummaryDummy2DB::RunSummaryDummy2DB( const std::string& dest):DataPipe(dest){}
  void RunSummaryDummy2DB::retrieveRun( unsigned int ){
  }
  const std::string RunSummaryDummy2DB::dataType() const{
    return "RUNSUMMARY";
  }
  const std::string RunSummaryDummy2DB::sourceType() const{
    return "DUMMY";
  }
  RunSummaryDummy2DB::~RunSummaryDummy2DB(){}
}//ns lumi
#include "RecoLuminosity/LumiProducer/interface/DataPipeFactory.h"
DEFINE_EDM_PLUGIN(lumi::DataPipeFactory,lumi::RunSummaryDummy2DB,"RunSummaryDummy2DB");
#endif
