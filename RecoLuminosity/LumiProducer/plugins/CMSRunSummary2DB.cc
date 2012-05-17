#ifndef RecoLuminosity_LumiProducer_CMSRunSummary2DB_h 
#define RecoLuminosity_LumiProducer_CMSRunSummary2DB_h 
#include "CoralBase/AttributeList.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/AttributeSpecification.h"
#include "CoralBase/Exception.h"
#include "CoralBase/TimeStamp.h"
#include "RelationalAccess/ConnectionService.h"
#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/ITransaction.h"
#include "RelationalAccess/ITypeConverter.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ICursor.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/IView.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/ITableDataEditor.h"
#include "RelationalAccess/IBulkOperation.h"
#include "RecoLuminosity/LumiProducer/interface/DataPipe.h"
#include "RecoLuminosity/LumiProducer/interface/LumiNames.h"
#include "RecoLuminosity/LumiProducer/interface/idDealer.h"
#include "RecoLuminosity/LumiProducer/interface/Exception.h"
#include "RecoLuminosity/LumiProducer/interface/DBConfig.h"
#include "RecoLuminosity/LumiProducer/interface/ConstantDef.h"
#include <iostream>
#include <sstream>
#include "RecoLuminosity/LumiProducer/interface/DataPipe.h"
#include "RecoLuminosity/LumiProducer/interface/LumiNames.h"
#include "RecoLuminosity/LumiProducer/interface/idDealer.h"
#include "RecoLuminosity/LumiProducer/interface/Exception.h"
#include "RecoLuminosity/LumiProducer/interface/DBConfig.h"
#include "RecoLuminosity/LumiProducer/interface/ConstantDef.h"
#include <string>
namespace lumi{
  class CMSRunSummary2DB : public DataPipe{
  public:
    CMSRunSummary2DB( const std::string& dest);
    virtual void retrieveData( unsigned int runnumber );
    virtual const std::string dataType() const;
    virtual const std::string sourceType() const;
    unsigned int str2int(const std::string& s) const;
    virtual ~CMSRunSummary2DB();
  private:
    struct cmsrunsum{
      std::string sequence;
      std::string hltkey;
      std::string fillnumber; //convert to number when write into lumi
      coral::TimeStamp startT;
      coral::TimeStamp stopT;
    };
  };//cl CMSRunSummary2DB
  //
  //implementation
  //
  CMSRunSummary2DB::CMSRunSummary2DB(const std::string& dest):DataPipe(dest){}
  void CMSRunSummary2DB::retrieveData( unsigned int runnumber){
    /**
       select distinct name from runsession_parameter
       sequence: select string_value from cms_runinfo.runsession_parameter where runnumber=129265 and name='CMS.LVL0:SEQ_NAME'
       hltkey: select string_value from cms_runinfo.runsession_parameter where runnumber=129265 and name='CMS.LVL0:HLT_KEY_DESCRIPTION';
       fillnumber: select string_value from cms_runinfo.runsession_parameter where runnumber=129265 and name='CMS.SCAL:FILLN' and rownum<=1;
       start/stop time:
       select time from cms_runinfo.runsession_parameter where runnumber=129265 and name='CMS.LVL0:START_TIME_T';
       select time from cms_runinfo.runsession_parameter where runnumber=129265 and name='CMS.LVL0:STOP_TIME_T';
    **/
    cmsrunsum result;
    std::string runinfoschema("CMS_RUNINFO");
    std::string runsessionParamTable("RUNSESSION_PARAMETER");
    coral::ConnectionService* svc=new coral::ConnectionService;
    lumi::DBConfig dbconf(*svc);
    if(!m_authpath.empty()){
      dbconf.setAuthentication(m_authpath);
    }
 
    //std::cout<<"m_source "<<m_source<<std::endl;
    coral::ISessionProxy* runinfosession=svc->connect(m_source,coral::ReadOnly);
    try{
      coral::ITypeConverter& tpc=runinfosession->typeConverter();
      tpc.setCppTypeForSqlType("unsigned int","NUMBER(38)");
      runinfosession->transaction().start(true);
      coral::ISchema& runinfoschemaHandle=runinfosession->schema(runinfoschema);
      if(!runinfoschemaHandle.existsTable(runsessionParamTable)){
	throw lumi::Exception(std::string("non-existing table "+runsessionParamTable),"CMSRunSummary2DB","retrieveData");
      }
      coral::IQuery* seqQuery=runinfoschemaHandle.tableHandle(runsessionParamTable).newQuery();
      coral::AttributeList seqBindVariableList;
      seqBindVariableList.extend("runnumber",typeid(unsigned int));
      seqBindVariableList.extend("name",typeid(std::string));
      
      seqBindVariableList["runnumber"].data<unsigned int>()=runnumber;
      seqBindVariableList["name"].data<std::string>()=std::string("CMS.LVL0:SEQ_NAME");
      seqQuery->setCondition("RUNNUMBER =:runnumber AND NAME =:name",seqBindVariableList);
      seqQuery->addToOutputList("STRING_VALUE");
      coral::ICursor& seqCursor=seqQuery->execute();
      
      while( seqCursor.next() ){
	const coral::AttributeList& row=seqCursor.currentRow();
	result.sequence=row["STRING_VALUE"].data<std::string>();
      }
      delete seqQuery;
      
      coral::IQuery* hltkeyQuery=runinfoschemaHandle.tableHandle(runsessionParamTable).newQuery();
      coral::AttributeList hltkeyBindVariableList;
      hltkeyBindVariableList.extend("runnumber",typeid(unsigned int));
      hltkeyBindVariableList.extend("name",typeid(std::string));

      hltkeyBindVariableList["runnumber"].data<unsigned int>()=runnumber;
      hltkeyBindVariableList["name"].data<std::string>()=std::string("CMS.LVL0:HLT_KEY_DESCRIPTION");
      hltkeyQuery->setCondition("RUNNUMBER =:runnumber AND NAME =:name",hltkeyBindVariableList);
      hltkeyQuery->addToOutputList("STRING_VALUE");   
      coral::ICursor& hltkeyCursor=hltkeyQuery->execute();
      
    while( hltkeyCursor.next() ){
      const coral::AttributeList& row=hltkeyCursor.currentRow();
      result.hltkey=row["STRING_VALUE"].data<std::string>();
    }
    delete hltkeyQuery;
    
    coral::IQuery* fillQuery=runinfoschemaHandle.tableHandle(runsessionParamTable).newQuery();
    coral::AttributeList fillBindVariableList;
    fillBindVariableList.extend("runnumber",typeid(unsigned int));
    fillBindVariableList.extend("name",typeid(std::string));

    fillBindVariableList["runnumber"].data<unsigned int>()=runnumber;
    fillBindVariableList["name"].data<std::string>()=std::string("CMS.SCAL:FILLN");
    fillQuery->setCondition("RUNNUMBER =:runnumber AND NAME =:name",fillBindVariableList);
    fillQuery->addToOutputList("STRING_VALUE"); 
    fillQuery->limitReturnedRows(1);
    coral::ICursor& fillCursor=fillQuery->execute();
    
    while( fillCursor.next() ){
      const coral::AttributeList& row=fillCursor.currentRow();
      result.fillnumber=row["STRING_VALUE"].data<std::string>();
    }
    delete fillQuery;
    if (result.fillnumber.empty()){
      throw nonCollisionException("retrieveData","CMSRunSummary2DB");
    }
    coral::IQuery* startTQuery=runinfoschemaHandle.tableHandle(runsessionParamTable).newQuery();
    coral::AttributeList startTVariableList;
    startTVariableList.extend("runnumber",typeid(unsigned int));
    startTVariableList.extend("name",typeid(std::string));

    startTVariableList["runnumber"].data<unsigned int>()=runnumber;
    startTVariableList["name"].data<std::string>()=std::string("CMS.LVL0:START_TIME_T");
    startTQuery->setCondition("RUNNUMBER =:runnumber AND NAME =:name",startTVariableList);
    startTQuery->addToOutputList("TIME"); 
    coral::ICursor& startTCursor=startTQuery->execute();
    
    while( startTCursor.next() ){
      const coral::AttributeList& row=startTCursor.currentRow();
      result.startT=row["TIME"].data<coral::TimeStamp>();
    }
    delete startTQuery;
    coral::IQuery* stopTQuery=runinfoschemaHandle.tableHandle(runsessionParamTable).newQuery();
    coral::AttributeList stopTVariableList;
    stopTVariableList.extend("runnumber",typeid(unsigned int));
    stopTVariableList.extend("name",typeid(std::string));

    stopTVariableList["runnumber"].data<unsigned int>()=runnumber;
    stopTVariableList["name"].data<std::string>()=std::string("CMS.LVL0:STOP_TIME_T");
    stopTQuery->setCondition("RUNNUMBER =:runnumber AND NAME =:name",stopTVariableList);
    stopTQuery->addToOutputList("TIME"); 
    coral::ICursor& stopTCursor=stopTQuery->execute();
    
    while( stopTCursor.next() ){
      const coral::AttributeList& row=stopTCursor.currentRow();
      result.stopT=row["TIME"].data<coral::TimeStamp>();
    }
    delete stopTQuery;
    }catch( const coral::Exception& er){
      runinfosession->transaction().rollback();
      delete runinfosession;
      delete svc;
      throw er;
    }
    runinfosession->transaction().commit();
    delete runinfosession;
    //std::cout<<"result for run "<<runnumber<<" : sequence : "<<result.sequence<<" : "<<result.hltkey<<" : hltkey : "<<result.hltkey<<" : fillnumber : "<<result.fillnumber<<std::endl; 

    //std::cout<<"connecting to dest "<<m_dest<<std::endl; 
    coral::ISessionProxy* destsession=svc->connect(m_dest,coral::Update);
    
    coral::ITypeConverter& desttpc=destsession->typeConverter();
    desttpc.setCppTypeForSqlType("unsigned int","NUMBER(10)");
    try{
      destsession->transaction().start(false);
      coral::ISchema& destschema=destsession->nominalSchema();
      coral::ITable& destruntable=destschema.tableHandle(LumiNames::cmsrunsummaryTableName());
      coral::AttributeList runData;
      destruntable.dataEditor().rowBuffer(runData);
      runData["RUNNUM"].data<unsigned int>()=runnumber;
      runData["FILLNUM"].data<unsigned int>()=str2int(result.fillnumber);
      runData["SEQUENCE"].data<std::string>()=result.sequence;
      runData["HLTKEY"].data<std::string>()=result.hltkey;
      runData["STARTTIME"].data<coral::TimeStamp>()=result.startT;
      runData["STOPTIME"].data<coral::TimeStamp>()=result.stopT;
      destruntable.dataEditor().insertRow(runData);
    }catch( const coral::Exception& er){
      std::cout<<"database problem "<<er.what()<<std::endl;
      destsession->transaction().rollback();
      delete destsession;
      delete svc;
      throw er;
    }
    destsession->transaction().commit();
    delete svc;
  }
  const std::string CMSRunSummary2DB::dataType() const{
    return "CMSRUNSUMMARY";
  }
  const std::string CMSRunSummary2DB::sourceType() const{
    return "DB";
  }
  unsigned int CMSRunSummary2DB::str2int(const std::string& s)  const{
    std::istringstream myStream(s);
    unsigned int i;
    if(myStream>>i){
      return i;
    }else{
      throw lumi::Exception(std::string("str2int error"),"str2int","CMSRunSummary2DB");
    }
  }
  CMSRunSummary2DB::~CMSRunSummary2DB(){}
}//ns lumi
#include "RecoLuminosity/LumiProducer/interface/DataPipeFactory.h"
DEFINE_EDM_PLUGIN(lumi::DataPipeFactory,lumi::CMSRunSummary2DB,"CMSRunSummary2DB");
#endif
