#ifndef RecoLuminosity_LumiProducer_CMSRunSummaryDummy2DB_H 
#define RecoLuminosity_LumiProducer_CMSRunSummaryDummy2DB_H 
#include "RelationalAccess/ConnectionService.h"
#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/ITransaction.h"
#include "RelationalAccess/ITypeConverter.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/ITableDataEditor.h"
#include "CoralBase/TimeStamp.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/AttributeSpecification.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/Exception.h"
#include "RecoLuminosity/LumiProducer/interface/DataPipe.h"
#include "RecoLuminosity/LumiProducer/interface/LumiNames.h"
#include "RecoLuminosity/LumiProducer/interface/Exception.h"
#include "RecoLuminosity/LumiProducer/interface/DBConfig.h"
#include <iostream>
namespace lumi{
  class CMSRunSummaryDummy2DB : public DataPipe{
  public:
    CMSRunSummaryDummy2DB(const std::string& dest);
    virtual unsigned long long retrieveData( unsigned int );
    virtual const std::string dataType() const;
    virtual const std::string sourceType() const;
    virtual ~CMSRunSummaryDummy2DB();
  };//cl CMSRunSummaryDummy2DB
  //
  //implementation
  //
  CMSRunSummaryDummy2DB::CMSRunSummaryDummy2DB( const std::string& dest):DataPipe(dest){}
  unsigned long long CMSRunSummaryDummy2DB::retrieveData( unsigned int runnum){
    //
    //generate dummy data of run summary for the given run and write data to LumiDB
    //
    std::string fakehltkey("/cdaq/Cosmic/V12");
    coral::ConnectionService* svc=new coral::ConnectionService;
    lumi::DBConfig dbconf(*svc);
    if(!m_authpath.empty()){
      dbconf.setAuthentication(m_authpath);
    }
    coral::ISessionProxy* session=svc->connect(m_dest,coral::Update);
    coral::ITypeConverter& tpc=session->typeConverter();
    tpc.setCppTypeForSqlType("unsigned int","NUMBER(10)");
    try{
      session->transaction().start(false);
      coral::ISchema& schema=session->nominalSchema();
      coral::ITable& runtable=schema.tableHandle(LumiNames::cmsrunsummaryTableName());
      coral::AttributeList runData;
      runtable.dataEditor().rowBuffer(runData);
      runData["RUNNUM"].data<unsigned int>()=runnum;
      runData["FILLNUM"].data<unsigned int>()=8973344;
      runData["SEQUENCE"].data<std::string>()="run sequence key";
      runData["HLTKEY"].data<std::string>()=fakehltkey;
      runData["STARTTIME"].data<coral::TimeStamp>()=coral::TimeStamp::now();
      runData["STOPTIME"].data<coral::TimeStamp>()=coral::TimeStamp::now();
      
      runtable.dataEditor().insertRow(runData);
    }catch( const coral::Exception& er){
      std::cout<<"database problem "<<er.what()<<std::endl;
      session->transaction().rollback();
      delete session;
      delete svc;
      throw er;
    }
    session->transaction().commit();
    delete session;
    delete svc;
    return 0;
  }
  const std::string CMSRunSummaryDummy2DB::dataType() const{
    return "CMSRUNSUMMARY";
  }
  const std::string CMSRunSummaryDummy2DB::sourceType() const{
    return "DUMMY";
  }
  CMSRunSummaryDummy2DB::~CMSRunSummaryDummy2DB(){}
}//ns lumi
#include "RecoLuminosity/LumiProducer/interface/DataPipeFactory.h"
DEFINE_EDM_PLUGIN(lumi::DataPipeFactory,lumi::CMSRunSummaryDummy2DB,"CMSRunSummaryDummy2DB");
#endif
