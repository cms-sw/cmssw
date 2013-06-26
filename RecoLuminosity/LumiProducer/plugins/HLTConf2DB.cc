#ifndef RecoLuminosity_LumiProducer_HLTConf2DB_H 
#define RecoLuminosity_LumiProducer_HLTConf2DB_H 

#include "CoralBase/AttributeList.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/AttributeSpecification.h"
#include "CoralBase/Exception.h"
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
#include <map>
#include <vector>
#include <string>
namespace lumi{
  class HLTConf2DB : public DataPipe{
  public:
    explicit HLTConf2DB( const std::string& dest);
    virtual unsigned long long retrieveData( unsigned int );
    virtual const std::string dataType() const;
    virtual const std::string sourceType() const;
    virtual ~HLTConf2DB();
  };//cl HLTConf2DB
  //
  //implementation
  //
  HLTConf2DB::HLTConf2DB( const std::string& dest):DataPipe(dest){}
  unsigned long long HLTConf2DB::retrieveData( unsigned int runnumber ){
    std::string runinfoschema("CMS_RUNINFO");
    //std::string hltschema("CMS_HLT_V0");
    std::string hltschema("CMS_HLT");
    std::string runsessiontable("RUNSESSION_PARAMETER");
    //
    //must login as cms_hlt_r
    coral::ConnectionService* svc=new coral::ConnectionService;
    lumi::DBConfig dbconf(*svc);
    if(!m_authpath.empty()){
      dbconf.setAuthentication(m_authpath);
    }
    //std::cout<<"m_source "<<m_source<<std::endl;
    coral::ISessionProxy* srcsession=svc->connect(m_source, coral::ReadOnly);
    coral::ITypeConverter& tpc=srcsession->typeConverter();
    tpc.setCppTypeForSqlType("unsigned int","NUMBER(38)");
    //
    //select string_value from CMS_RUNINFO.runsession_parameter where name='CMS.LVL0:HLT_KEY_DESCRIPTION' and runnumber=xx;
    //
    std::string hltkey("");
    std::vector< std::pair<std::string,std::string> > hlt2l1map;
    hlt2l1map.reserve(200);
    try{
      srcsession->transaction().start(true);
      coral::ISchema& runinfoSchemaHandle=srcsession->schema(runinfoschema);
      if( !runinfoSchemaHandle.existsTable(runsessiontable) ){
	throw lumi::Exception("missing runsession table","retrieveData","HLTConf2DB");
      }
      coral::AttributeList rfBindVariableList;
      rfBindVariableList.extend("name",typeid(std::string));
      rfBindVariableList.extend("runnumber",typeid(unsigned int));
      rfBindVariableList["name"].data<std::string>()=std::string("CMS.LVL0:HLT_KEY_DESCRIPTION");
      rfBindVariableList["runnumber"].data<unsigned int>()=runnumber;
      coral::IQuery* kq=runinfoSchemaHandle.newQuery();
      coral::AttributeList runinfoOut;
      runinfoOut.extend("STRING_VALUE",typeid(std::string));
      kq->addToTableList(runsessiontable);
      kq->setCondition("NAME = :name AND RUNNUMBER = :runnumber",rfBindVariableList);
      kq->addToOutputList("STRING_VALUE");
      kq->defineOutput(runinfoOut);
      coral::ICursor& kcursor=kq->execute();
      unsigned int s=0;
      while( kcursor.next() ){
	const coral::AttributeList& row=kcursor.currentRow();
	hltkey=row["STRING_VALUE"].data<std::string>();
	++s;
      }
      if( s==0||hltkey.empty() ){
	kcursor.close();
	delete kq;
	srcsession->transaction().rollback();
	throw lumi::Exception(std::string("requested run has no or invalid hltkey"),"retrieveData","TRG2DB");
      }
      if(s>1){
	kcursor.close();
	delete kq;
	srcsession->transaction().rollback();
	throw lumi::Exception(std::string("requested run has multile hltkey"),"retrieveData","TRG2DB");
      }
    }catch( const coral::Exception& er ){
      std::cout<<"source runinfo database problem "<<er.what()<<std::endl;
      srcsession->transaction().rollback();
      delete srcsession;
      delete svc;
      throw er;
    }
    coral::ISchema& confSchemaHandle=srcsession->nominalSchema();
    try{
      //srcsession->transaction().start(true);
      if( !confSchemaHandle.existsTable("PATHS") || !confSchemaHandle.existsTable("STRINGPARAMVALUES") || !confSchemaHandle.existsTable("PARAMETERS") || !confSchemaHandle.existsTable("SUPERIDPARAMETERASSOC") || !confSchemaHandle.existsTable("MODULES") || !confSchemaHandle.existsTable("MODULETEMPLATES") || !confSchemaHandle.existsTable("PATHMODULEASSOC") || !confSchemaHandle.existsTable("CONFIGURATIONPATHASSOC") || !confSchemaHandle.existsTable("CONFIGURATIONS") ){
	throw lumi::Exception("missing hlt conf tables" ,"retrieveData","HLTConf2DB");
      }
      coral::AttributeList q1BindVariableList;
      q1BindVariableList.extend("hltseed",typeid(std::string));
      q1BindVariableList.extend("l1seedexpr",typeid(std::string));
      q1BindVariableList.extend("hltkey",typeid(std::string));
      q1BindVariableList["hltseed"].data<std::string>()=std::string("HLTLevel1GTSeed");
      q1BindVariableList["l1seedexpr"].data<std::string>()=std::string("L1SeedsLogicalExpression");
      q1BindVariableList["hltkey"].data<std::string>()=hltkey;    
      coral::IQuery* q1=confSchemaHandle.newQuery();
      q1->addToOutputList("PATHS.NAME","hltpath");
      q1->addToOutputList("STRINGPARAMVALUES.VALUE","l1expression");
      
      q1->addToTableList("PATHS");    
      q1->addToTableList("STRINGPARAMVALUES");
      q1->addToTableList("PARAMETERS");
      q1->addToTableList("SUPERIDPARAMETERASSOC");
      q1->addToTableList("MODULES");
      q1->addToTableList("MODULETEMPLATES");
      q1->addToTableList("PATHMODULEASSOC");
      q1->addToTableList("CONFIGURATIONPATHASSOC");
      q1->addToTableList("CONFIGURATIONS");
      
      q1->setCondition("PARAMETERS.PARAMID=STRINGPARAMVALUES.PARAMID and SUPERIDPARAMETERASSOC.PARAMID=PARAMETERS.PARAMID and MODULES.SUPERID=SUPERIDPARAMETERASSOC.SUPERID and MODULETEMPLATES.SUPERID=MODULES.TEMPLATEID and PATHMODULEASSOC.MODULEID=MODULES.SUPERID and PATHS.PATHID=PATHMODULEASSOC.PATHID and CONFIGURATIONPATHASSOC.PATHID=PATHS.PATHID and CONFIGURATIONS.CONFIGID=CONFIGURATIONPATHASSOC.CONFIGID and MODULETEMPLATES.NAME = :hltseed and PARAMETERS.NAME = :l1seedexpr and CONFIGURATIONS.CONFIGDESCRIPTOR = :hltkey",q1BindVariableList);

      /**
	 select paths.name,stringparamvalues.value from stringparamvalues,paths,parameters,superidparameterassoc,modules,moduletemplates,pathmoduleassoc,configurationpathassoc,configurations where parameters.paramid=stringparamvalues.paramid and  superidparameterassoc.paramid=parameters.paramid and modules.superid=superidparameterassoc.superid and moduletemplates.superid=modules.templateid and pathmoduleassoc.moduleid=modules.superid and paths.pathid=pathmoduleassoc.pathid and configurationpathassoc.pathid=paths.pathid and configurations.configid=configurationpathassoc.configid and moduletemplates.name='HLTLevel1GTSeed' and parameters.name='L1SeedsLogicalExpression' and configurations.configid=1905; 
      **/
      coral::ICursor& cursor1=q1->execute();
      while( cursor1.next() ){
	const coral::AttributeList& row=cursor1.currentRow();
	std::string hltpath=row["hltpath"].data<std::string>();
	std::string l1expression=row["l1expression"].data<std::string>();
	hlt2l1map.push_back(std::make_pair(hltpath,l1expression));
      }
      cursor1.close();
      delete q1;
    }catch( const coral::Exception& er ){
      std::cout<<"database problem with source hlt confdb"<<er.what()<<std::endl;
      srcsession->transaction().rollback();
      delete srcsession;
      throw er;
    }
    srcsession->transaction().commit();
    delete srcsession;
    std::vector< std::pair<std::string,std::string> >::const_iterator mIt;
    std::vector< std::pair<std::string,std::string> >::const_iterator mBeg=hlt2l1map.begin();
    std::vector< std::pair<std::string,std::string> >::const_iterator mEnd=hlt2l1map.end();
    
    coral::ISessionProxy* destsession=svc->connect(m_dest, coral::Update);
    try{      
      //check if hltkey already exists
      destsession->transaction().start(true);
      bool hltkeyExists=false;
      coral::AttributeList kQueryBindList;
      kQueryBindList.extend("hltkey",typeid(std::string));
      coral::IQuery* kQuery=destsession->nominalSchema().tableHandle(LumiNames::trghltMapTableName()).newQuery();
      kQuery->setCondition("HLTKEY =:hltkey",kQueryBindList);
      kQueryBindList["hltkey"].data<std::string>()=hltkey;
      coral::ICursor& kResult=kQuery->execute();
      while( kResult.next() ){
	hltkeyExists=true;
      }
      if(hltkeyExists){
	std::cout<<"hltkey "<<hltkey<<" already registered , do nothing"<<std::endl;
	destsession->transaction().commit();
	delete kQuery;
	delete svc;
	return 0;
      }
      destsession->transaction().commit();
      destsession->transaction().start(false);
      coral::ISchema& destschema=destsession->nominalSchema();
      coral::ITable& hltconftable=destschema.tableHandle(LumiNames::trghltMapTableName());
      coral::AttributeList hltconfData;
      hltconfData.extend<std::string>("HLTKEY");
      hltconfData.extend<std::string>("HLTPATHNAME");
      hltconfData.extend<std::string>("L1SEED");
      coral::IBulkOperation* hltconfInserter=hltconftable.dataEditor().bulkInsert(hltconfData,200);
      hltconfData["HLTKEY"].data<std::string>()=hltkey;
      std::string& hltpathname=hltconfData["HLTPATHNAME"].data<std::string>();
      std::string& l1seedname=hltconfData["L1SEED"].data<std::string>();
      for(mIt=mBeg; mIt!=mEnd; ++mIt ){
	hltpathname=mIt->first;
	l1seedname=mIt->second;
	hltconfInserter->processNextIteration();
	//std::cout<<mIt->first<<" "<<mIt->second<<std::endl;      
      }
      hltconfInserter->flush();
      delete hltconfInserter;
      destsession->transaction().commit();
    }catch( const coral::Exception& er){
      std::cout<<"database problem "<<er.what()<<std::endl;
      destsession->transaction().rollback();
      delete destsession;
      delete svc;
      throw er;
    }
    delete destsession;
    delete svc;
    return 0;
  }
  const std::string HLTConf2DB::dataType() const{
    return "HLTConf";
  }
  const std::string HLTConf2DB::sourceType() const{
    return "DB";
  }
  HLTConf2DB::~HLTConf2DB(){}
}//ns lumi
#include "RecoLuminosity/LumiProducer/interface/DataPipeFactory.h"
DEFINE_EDM_PLUGIN(lumi::DataPipeFactory,lumi::HLTConf2DB,"HLTConf2DB");
#endif
