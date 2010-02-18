#ifndef RecoLuminosity_LumiProducer_HLTConfDummy2DB_h 
#define RecoLuminosity_LumiProducer_HLTConfDummy2DB_h 
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
#include <iostream>
namespace lumi{
  class HLTConfDummy2DB : public DataPipe{
  public:
    explicit HLTConfDummy2DB(const std::string& dest);
    virtual void retrieveData( unsigned int hltconfigId);
    virtual const std::string dataType() const;
    virtual const std::string sourceType() const;
    virtual ~HLTConfDummy2DB();
  };//cl HLTConfDummy2DB
  //
  //implementation
  //
  HLTConfDummy2DB::HLTConfDummy2DB(const std::string& dest):DataPipe(dest){}
  void HLTConfDummy2DB::retrieveData( unsigned int hltconfigId){
    //
    //generate dummy configuration data for the given hltconfid and write data to LumiDB
    //
    coral::ConnectionService* svc=new coral::ConnectionService;
    coral::ISessionProxy* session=svc->connect(m_dest,coral::Update);
    try{
      unsigned int totalhltpath=126;
      session->transaction().start(false);
      coral::ISchema& schema=session->nominalSchema();
      coral::ITable& hltconftable=schema.tableHandle(LumiNames::trghltMapTableName());
      coral::AttributeList hltconfData;
      hltconfData.extend<unsigned int>("HLTCONFID");
      hltconfData.extend<std::string>("HLTPATHNAME");
      hltconfData.extend<std::string>("L1SEED");
      coral::IBulkOperation* hltconfInserter=hltconftable.dataEditor().bulkInsert(hltconfData,200);
      //
      //loop over hltpaths
      //
      hltconfData["HLTCONFID"].data<unsigned int>()=hltconfigId;
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
