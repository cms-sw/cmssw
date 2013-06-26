#ifndef RecoLuminosity_LumiProducer_HLTDummy2DB_h 
#define RecoLuminosity_LumiProducer_HLTDummy2DB_h 
#include "RelationalAccess/ConnectionService.h"
#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/ITransaction.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/ITableDataEditor.h"
#include "RelationalAccess/IBulkOperation.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/AttributeSpecification.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/Exception.h"
#include "RecoLuminosity/LumiProducer/interface/DataPipe.h"
#include "RecoLuminosity/LumiProducer/interface/LumiNames.h"
#include "RecoLuminosity/LumiProducer/interface/idDealer.h"
#include "RecoLuminosity/LumiProducer/interface/Exception.h"
#include "RecoLuminosity/LumiProducer/interface/DBConfig.h"
#include <iostream>
#include <cstdio>
namespace lumi{
  class HLTDummy2DB : public DataPipe{
  public:
    HLTDummy2DB( const std::string& dest);
    virtual unsigned long long retrieveData( unsigned int );
    virtual const std::string dataType() const;
    virtual const std::string sourceType() const;
    virtual ~HLTDummy2DB();
  };//cl HLTDummy2DB
  //
  //implementation
  //
  HLTDummy2DB::HLTDummy2DB(const std::string& dest):DataPipe(dest){}
  unsigned long long HLTDummy2DB::retrieveData( unsigned int runnum){
    //
    //generate dummy data of hlt for the given run and write data to LumiDB
    //
    coral::ConnectionService* svc=new coral::ConnectionService;
    lumi::DBConfig dbconf(*svc);
    if(!m_authpath.empty()){
      dbconf.setAuthentication(m_authpath);
    }
    coral::ISessionProxy* session=svc->connect(m_dest,coral::Update);
    try{
      unsigned int totalcmsls=32;
      session->transaction().start(false);
      coral::ISchema& schema=session->nominalSchema();
      lumi::idDealer idg(schema);
      coral::ITable& hlttable=schema.tableHandle(LumiNames::hltTableName());
      coral::AttributeList hltData;
      hltData.extend<unsigned long long>("HLT_ID");
      hltData.extend<unsigned int>("RUNNUM");
      hltData.extend<unsigned int>("CMSLSNUM");
      hltData.extend<std::string>("PATHNAME");
      hltData.extend<unsigned long long>("INPUTCOUNT");
      hltData.extend<unsigned long long>("ACCEPTCOUNT");
      hltData.extend<unsigned int>("PRESCALE");
      coral::IBulkOperation* hltInserter=hlttable.dataEditor().bulkInsert(hltData,totalcmsls*260);
      //loop over lumi LS
      unsigned long long& hlt_id=hltData["HLT_ID"].data<unsigned long long>();
      unsigned int& hltrunnum=hltData["RUNNUM"].data<unsigned int>();
      unsigned int& cmslsnum=hltData["CMSLSNUM"].data<unsigned int>();
      std::string& pathname=hltData["PATHNAME"].data<std::string>();
      unsigned long long& inputcount=hltData["INPUTCOUNT"].data<unsigned long long>();
      unsigned long long& acceptcount=hltData["ACCEPTCOUNT"].data<unsigned long long>();
      unsigned int& prescale=hltData["PRESCALE"].data<unsigned int>();
      
      for(unsigned int i=1;i<=totalcmsls;++i){
	for(unsigned int j=1;j<165;++j){
	  hlt_id = idg.generateNextIDForTable(LumiNames::hltTableName());
	  hltrunnum = runnum;
	  cmslsnum = i;
	  char c[10];
	  ::sprintf(c,"%d",j);
	  pathname=std::string("FakeHLTPath_")+std::string(c);
	  inputcount=12345;
	  acceptcount=10;
	  prescale=1;
	  hltInserter->processNextIteration();
	}
      }
      hltInserter->flush();
      delete hltInserter;
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
    return 0;
  }
  const std::string HLTDummy2DB::dataType() const{
    return "HLT";
  }
  const std::string HLTDummy2DB::sourceType() const{
    return "DUMMY";
  }
  HLTDummy2DB::~HLTDummy2DB(){}
}//ns lumi
#include "RecoLuminosity/LumiProducer/interface/DataPipeFactory.h"
DEFINE_EDM_PLUGIN(lumi::DataPipeFactory,lumi::HLTDummy2DB,"HLTDummy2DB");
#endif
