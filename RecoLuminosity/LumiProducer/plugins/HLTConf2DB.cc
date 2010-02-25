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
    virtual void retrieveData( unsigned int );
    virtual const std::string dataType() const;
    virtual const std::string sourceType() const;
    virtual ~HLTConf2DB();
  };//cl HLTConf2DB
  //
  //implementation
  //
  HLTConf2DB::HLTConf2DB( const std::string& dest):DataPipe(dest){}
  void HLTConf2DB::retrieveData( unsigned int configid ){
    //must login as cms_hlt_r
    //select HLTKEY from HLT_SUPERVISOR_SCALAR_MAP where RUNNR=124025
    std::string runinfoschema("CMS_RUNINFO");
    std::string hltschema("CMS_HLT");
    coral::ConnectionService* svc=new coral::ConnectionService;
    lumi::DBConfig dbconf(*svc);
    if(!m_authpath.empty()){
      dbconf.setAuthentication(m_authpath);
    }
    //std::cout<<"m_source "<<m_source<<std::endl;
    coral::ISessionProxy* srcsession=svc->connect(m_source, coral::ReadOnly);
    //coral::ITypeConverter& tpc=srcsession->typeConverter();
    //unsigned int configid=1905;
    srcsession->transaction().start(true);
    coral::ISchema& confSchemaHandle=srcsession->schema(hltschema);
    if( !confSchemaHandle.existsTable("PATHS") || !confSchemaHandle.existsTable("STRINGPARAMVALUES") || !confSchemaHandle.existsTable("PARAMETERS") || !confSchemaHandle.existsTable("SUPERIDPARAMETERASSOC") || !confSchemaHandle.existsTable("MODULES") || !confSchemaHandle.existsTable("MODULETEMPLATES") || !confSchemaHandle.existsTable("PATHMODULEASSOC") || !confSchemaHandle.existsTable("CONFIGURATIONPATHASSOC") || !confSchemaHandle.existsTable("CONFIGURATIONS") ){
      throw lumi::Exception("missing hlt conf tables" ,"retrieveData","HLTConf2DB");
    }
    coral::AttributeList q1BindVariableList;
    q1BindVariableList.extend("hltseed",typeid(std::string));
    q1BindVariableList.extend("l1seedexpr",typeid(std::string));
    q1BindVariableList.extend("hltkey",typeid(unsigned int));
    q1BindVariableList["hltseed"].data<std::string>()=std::string("HLTLevel1GTSeed");
    q1BindVariableList["l1seedexpr"].data<std::string>()=std::string("L1SeedsLogicalExpression");
    q1BindVariableList["hltkey"].data<unsigned int>()=configid;    
    
    coral::IQuery* q1=srcsession->nominalSchema().newQuery();

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
    
    q1->setCondition("PARAMETERS.PARAMID=STRINGPARAMVALUES.PARAMID and SUPERIDPARAMETERASSOC.PARAMID=PARAMETERS.PARAMID and MODULES.SUPERID=SUPERIDPARAMETERASSOC.SUPERID and MODULETEMPLATES.SUPERID=MODULES.TEMPLATEID and PATHMODULEASSOC.MODULEID=MODULES.SUPERID and PATHS.PATHID=PATHMODULEASSOC.PATHID and CONFIGURATIONPATHASSOC.PATHID=PATHS.PATHID and CONFIGURATIONS.CONFIGID=CONFIGURATIONPATHASSOC.CONFIGID and MODULETEMPLATES.NAME = :hltseed and PARAMETERS.NAME = :l1seedexpr and CONFIGURATIONS.CONFIGID = :hltkey",q1BindVariableList);
    
    /**
       select paths.name,stringparamvalues.value from stringparamvalues,paths,parameters,superidparameterassoc,modules,moduletemplates,pathmoduleassoc,configurationpathassoc,configurations where parameters.paramid=stringparamvalues.paramid and  superidparameterassoc.paramid=parameters.paramid and modules.superid=superidparameterassoc.superid and moduletemplates.superid=modules.templateid and pathmoduleassoc.moduleid=modules.superid and paths.pathid=pathmoduleassoc.pathid and configurationpathassoc.pathid=paths.pathid and configurations.configid=configurationpathassoc.configid and moduletemplates.name='HLTLevel1GTSeed' and parameters.name='L1SeedsLogicalExpression' and configurations.configid=1905; 
    **/
    std::vector< std::pair<std::string,std::string> > hlt2l1map;
    hlt2l1map.reserve(200);
    coral::ICursor& cursor1=q1->execute();
    while( cursor1.next() ){
      const coral::AttributeList& row=cursor1.currentRow();
      std::string hltpath=row["hltpath"].data<std::string>();
      std::string l1expression=row["l1expression"].data<std::string>();
      hlt2l1map.push_back(std::make_pair(hltpath,l1expression));
    }
    cursor1.close();
    delete q1;
    srcsession->transaction().commit();
    delete srcsession;
    
    std::vector< std::pair<std::string,std::string> >::const_iterator mIt;
    std::vector< std::pair<std::string,std::string> >::const_iterator mBeg=hlt2l1map.begin();
    std::vector< std::pair<std::string,std::string> >::const_iterator mEnd=hlt2l1map.end();
    
    coral::ISessionProxy* destsession=svc->connect(m_dest, coral::Update);
    try{
      destsession->transaction().start(false);
      coral::ISchema& destschema=destsession->nominalSchema();
      coral::ITable& hltconftable=destschema.tableHandle(LumiNames::trghltMapTableName());
      coral::AttributeList hltconfData;
      hltconfData.extend<unsigned int>("HLTCONFID");
      hltconfData.extend<std::string>("HLTPATHNAME");
      hltconfData.extend<std::string>("L1SEED");
      coral::IBulkOperation* hltconfInserter=hltconftable.dataEditor().bulkInsert(hltconfData,200);
      hltconfData["HLTCONFID"].data<unsigned int>()=configid;
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
    }catch( const coral::Exception& er){
      std::cout<<"database problem "<<er.what()<<std::endl;
      destsession->transaction().rollback();
      delete destsession;
      delete svc;
      throw er;
    }
    destsession->transaction().commit();
    delete destsession;
    delete svc;
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
