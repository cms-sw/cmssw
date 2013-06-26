#ifndef RecoLuminosity_LumiProducer_HLTConfDummy2DB_h 
#define RecoLuminosity_LumiProducer_HLTConfDummy2DB_h 
#include "RelationalAccess/ConnectionService.h"
#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/ITransaction.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ICursor.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/ITableDataEditor.h"
#include "RelationalAccess/IBulkOperation.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/AttributeSpecification.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/Exception.h"
#include "RecoLuminosity/LumiProducer/interface/DataPipe.h"
#include "RecoLuminosity/LumiProducer/interface/LumiNames.h"
#include "RecoLuminosity/LumiProducer/interface/Exception.h"
#include "RecoLuminosity/LumiProducer/interface/DBConfig.h"
#include <iostream>
#include <cstdio>

namespace lumi{
  class HLTConfDummy2DB : public DataPipe{
  public:
    explicit HLTConfDummy2DB(const std::string& dest);
    virtual unsigned long long retrieveData( unsigned int runnumber);
    virtual const std::string dataType() const;
    virtual const std::string sourceType() const;
    virtual ~HLTConfDummy2DB();
  };//cl HLTConfDummy2DB
  //
  //implementation
  //
  HLTConfDummy2DB::HLTConfDummy2DB(const std::string& dest):DataPipe(dest){}
  unsigned long long HLTConfDummy2DB::retrieveData( unsigned int runnumber){
    //
    //generate dummy configuration data for the given hltconfid and write data to LumiDB
    //
    //std::cout<<"retrieving data for run "<<runnumber<<std::endl;
    std::string fakehltkey("/cdaq/Cosmic/V12");
    coral::ConnectionService* svc=new coral::ConnectionService;
    lumi::DBConfig dbconf(*svc);
    if(!m_authpath.empty()){
      dbconf.setAuthentication(m_authpath);
    }
    coral::ISessionProxy* session=svc->connect(m_dest,coral::Update);
    try{
      unsigned int totalhltpath=126;
      session->transaction().start(false);
      coral::ISchema& schema=session->nominalSchema();
      coral::ITable& hltconftable=schema.tableHandle(LumiNames::trghltMapTableName());
      coral::AttributeList hltconfData;
      hltconfData.extend<std::string>("HLTKEY");
      hltconfData.extend<std::string>("HLTPATHNAME");
      hltconfData.extend<std::string>("L1SEED");
      coral::IBulkOperation* hltconfInserter=hltconftable.dataEditor().bulkInsert(hltconfData,200);
      //
      //loop over hltpaths
      //
      hltconfData["HLTKEY"].data<std::string>()=fakehltkey;
      std::string& hltpathname=hltconfData["HLTPATHNAME"].data<std::string>();
      std::string& l1seed=hltconfData["L1SEED"].data<std::string>();
      for(unsigned int i=1;i<=totalhltpath;++i){
	char c[10];
	::sprintf(c,"-%d",i);
	hltpathname=std::string("FakeHLTPATH_")+std::string(c);
	l1seed=std::string("FakeL1SeedExpression_")+std::string(c);
	hltconfInserter->processNextIteration();
      }
      hltconfInserter->flush();
      delete hltconfInserter;
      session->transaction().commit();
    }catch( const coral::Exception& er){
      std::cout<<"database problem "<<er.what()<<std::endl;
      session->transaction().rollback();
      delete session;
      delete svc;
      throw er;
    }
    delete session;
    delete svc;
    return 0;
  }
  const std::string HLTConfDummy2DB::dataType() const{
    return "HLTCONF";
  }
  const std::string HLTConfDummy2DB::sourceType() const{
    return "DUMMY";
  }
  HLTConfDummy2DB::~HLTConfDummy2DB(){}
}//ns lumi
#include "RecoLuminosity/LumiProducer/interface/DataPipeFactory.h"
DEFINE_EDM_PLUGIN(lumi::DataPipeFactory,lumi::HLTConfDummy2DB,"HLTConfDummy2DB");
#endif
