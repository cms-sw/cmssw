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
namespace lumi{
  class HLTConfDummy2DB : public DataPipe{
  public:
    explicit HLTConfDummy2DB(const std::string& dest);
    virtual void retrieveData( unsigned int runnumber);
    virtual const std::string dataType() const;
    virtual const std::string sourceType() const;
    virtual ~HLTConfDummy2DB();
  };//cl HLTConfDummy2DB
  //
  //implementation
  //
  HLTConfDummy2DB::HLTConfDummy2DB(const std::string& dest):DataPipe(dest){}
  void HLTConfDummy2DB::retrieveData( unsigned int runnumber){
    //
    //generate dummy configuration data for the given hltconfid and write data to LumiDB
    //
    std::string fakehltkey("/cdaq/Cosmic/V12");
    coral::ConnectionService* svc=new coral::ConnectionService;
    lumi::DBConfig dbconf(*svc);
    if(!m_authpath.empty()){
      dbconf.setAuthentication(m_authpath);
    }
    coral::ISessionProxy* session=svc->connect(m_dest,coral::Update);
    try{
      unsigned int totalhltpath=126;
      //session->transaction().start(true);
      //
      //check if this hltkey is also registered in runsummary table
      //
      /**coral::AttributeList rqBindVariables;
	 rqBindVariables.extend("runnumber",typeid(unsigned int));
	 rqBindVariables["runnumber"].data<unsigned int>()=runnumber;
	 coral::AttributeList rqResult;
	 rqResult.extend("HLTKEY",typeid(std::string));
	 coral::IQuery* rq=session->nominalSchema().tableHandle(LumiNames::runsummaryTableName()).newQuery();
	 rq->addToOutputList("HLTKEY");
	 rq->setCondition("RUNNUM = :runnumber",rqBindVariables);
	 rq->defineOutput(rqResult);
	 coral::ICursor& rqCursor=rq->execute();
	 std::string hltkey;
	 unsigned int s=0;
	 while( rqCursor.next() ){
	 const coral::AttributeList& row=rqCursor.currentRow();
	 hltkey=row["HLTKEY"].data<std::string>();
	 ++s;
	 }
	 if(s==0 || hltkey.empty()){
	 std::cout<<"hltkey is not registered for requested run"<<std::endl;
	 writeKeyToSunSummary=true;
	 }
	 if(s>1){
	 throw lumi::Exception("hltkey is not unique for the requested run","retrieveData","HLTConfDummy2DB");
	 }
	 if( s!=0 && hltkey!=fakehltkey ){
	 throw lumi::Exception("exist a different hltkey for the requested run","retrieveData","HLTConfDummy2DB");
	 }
	 session->transaction().commit();
	 delete rq;
      **/
      session->transaction().start(false);
      coral::ISchema& schema=session->nominalSchema();
      coral::ITable& hltconftable=schema.tableHandle(LumiNames::trghltMapTableName());
      coral::AttributeList hltconfData;
      hltconfData.extend<unsigned int>("RUNNUM");
      hltconfData.extend<std::string>("HLTKEY");
      hltconfData.extend<std::string>("HLTPATHNAME");
      hltconfData.extend<std::string>("L1SEED");
      coral::IBulkOperation* hltconfInserter=hltconftable.dataEditor().bulkInsert(hltconfData,200);
      //
      //loop over hltpaths
      //
      hltconfData["HLTKEY"].data<std::string>()=fakehltkey;
      hltconfData["RUNNUM"].data<unsigned int>()=runnumber;
      std::string& hltpathname=hltconfData["HLTPATHNAME"].data<std::string>();
      std::string& l1seed=hltconfData["L1SEED"].data<std::string>();
      for(unsigned int i=1;i<=totalhltpath;++i){
	char c[10];
	::sprintf(c,"-%d",i);
	hltpathname=std::string("FakeHLTPATH_")+std::string(c);
	l1seed==std::string("FakeL1SeedExpression_")+std::string(c);
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
