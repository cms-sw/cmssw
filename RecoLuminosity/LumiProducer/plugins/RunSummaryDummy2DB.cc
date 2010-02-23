#ifndef RecoLuminosity_LumiProducer_RunSummaryDummy2DB_h 
#define RecoLuminosity_LumiProducer_RunSummaryDummy2DB_h 
#include "RelationalAccess/ConnectionService.h"
#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/ITransaction.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/ITableDataEditor.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/AttributeSpecification.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/Exception.h"
#include "RecoLuminosity/LumiProducer/interface/DataPipe.h"
#include "RecoLuminosity/LumiProducer/interface/LumiNames.h"
#include <iostream>
#include "RecoLuminosity/LumiProducer/interface/DataPipe.h"
namespace lumi{
  class RunSummaryDummy2DB : public DataPipe{
  public:
    RunSummaryDummy2DB(const std::string& dest);
    virtual void retrieveData( unsigned int );
    virtual const std::string dataType() const;
    virtual const std::string sourceType() const;
    virtual ~RunSummaryDummy2DB();
  };//cl RunSummaryDummy2DB
  //
  //implementation
  //
  RunSummaryDummy2DB::RunSummaryDummy2DB( const std::string& dest):DataPipe(dest){}
  void RunSummaryDummy2DB::retrieveData( unsigned int runnum){
    //
    //generate dummy data of run summary for the given run and write data to LumiDB
    //
    coral::ConnectionService* svc=new coral::ConnectionService;
    coral::ISessionProxy* session=svc->connect(m_dest,coral::Update);
    try{
      session->transaction().start(false);
      coral::ISchema& schema=session->nominalSchema();
      coral::ITable& runtable=schema.tableHandle(LumiNames::runsummaryTableName());
      coral::AttributeList runData;
      runtable.dataEditor().rowBuffer(runData);
      runData["RUNNUM"].data<unsigned int>()=runnum;
      runData["FILLNUM"].data<unsigned int>()=8973344;
      runData["TOTALCMSLS"].data<unsigned int>()=32;
      runData["TOTALLUMILS"].data<unsigned int>()=35;
      runData["SEQUENCE"].data<std::string>()="run sequence key";
      runData["HLTCONFIGID"].data<unsigned int>()=6785;
      runData["STARTORBIT"].data<unsigned int>()=340506;
      runData["ENDORBIT"].data<unsigned int>()=7698988;
      runtable.dataEditor().insertRow(runData);
    }catch( const coral::Exception& er){
      std::cout<<"database problem "<<er.what()<<std::endl;
      session->transaction().rollback();
      delete session;
      delete svc;
      throw er;
    }
    //delete detailInserter;
    session->transaction().commit();
    delete session;
    delete svc;
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
