#ifndef RecoLuminosity_LumiProducer_TRGDummy2DB_h 
#define RecoLuminosity_LumiProducer_TRGDummy2DB_h 
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
#include <iostream>
#include <cstdio>
namespace lumi{
  class TRGDummy2DB : public DataPipe{
  public:
    TRGDummy2DB(const std::string& dest);
    virtual void retrieveData( unsigned int );
    virtual const std::string dataType() const;
    virtual const std::string sourceType() const;
    virtual ~TRGDummy2DB();
  };//cl TRGDummy2DB
  //
  //implementation
  //
  TRGDummy2DB::TRGDummy2DB(const std::string& dest):DataPipe(dest){}
  void TRGDummy2DB::retrieveData( unsigned int runnum){
    //
    //generate dummy data of trg for the given run and write data to LumiDB
    //
     coral::ConnectionService* svc=new coral::ConnectionService;
    coral::ISessionProxy* session=svc->connect(m_dest,coral::Update);
    try{
      unsigned int totalcmsls=32;
      session->transaction().start(false);
      coral::ISchema& schema=session->nominalSchema();
      lumi::idDealer idg(schema);
      coral::ITable& trgtable=schema.tableHandle(LumiNames::trgTableName());
      coral::AttributeList trgData;
      trgData.extend<unsigned long long>("TRG_ID");
      trgData.extend<unsigned int>("RUNNUM");
      trgData.extend<unsigned int>("CMSLUMINUM");
      trgData.extend<unsigned int>("BITNUM");
      trgData.extend<std::string>("BITNAME");
      trgData.extend<unsigned long long>("COUNT");
      trgData.extend<unsigned long long>("DEADTIME");
      trgData.extend<unsigned int>("PRESCALE");
      coral::IBulkOperation* trgInserter=trgtable.dataEditor().bulkInsert(trgData,totalcmsls*192);
      //loop over lumi LS
      unsigned long long& trg_id=trgData["TRG_ID"].data<unsigned long long>();
      unsigned int& trgrunnum=trgData["RUNNUM"].data<unsigned int>();
      unsigned int& cmsluminum=trgData["CMSLUMINUM"].data<unsigned int>();
      unsigned int& bitnum=trgData["BITNUM"].data<unsigned int>();
      std::string& bitname=trgData["BITNAME"].data<std::string>();
      unsigned long long& count=trgData["COUNT"].data<unsigned long long>();
      unsigned long long& deadtime=trgData["DEADTIME"].data<unsigned long long>();
      unsigned int& prescale=trgData["PRESCALE"].data<unsigned int>();
      
      for(unsigned int i=1;i<=totalcmsls;++i){
	for(unsigned int j=0;j<192;++j){ //total n of trg bits
	  trg_id = idg.generateNextIDForTable(LumiNames::trgTableName());
	  trgrunnum = runnum;
	  cmsluminum = i;
	  bitnum=j;
	  char c[10];
	  if(j>127){
	    ::sprintf(c,"%d",j-127);
	    bitname=std::string(c);
	  }else{
	    ::sprintf(c,"%d",j);
	    bitname=std::string("FakeTRGBIT_")+std::string(c);
	  }
	  count=12345;
	  deadtime=6785;
	  prescale=1;
	  trgInserter->processNextIteration();
	}
      }
      trgInserter->flush();
      delete trgInserter;
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
  const std::string TRGDummy2DB::dataType() const{
    return "TRG";
  }
  const std::string TRGDummy2DB::sourceType() const{
    return "DUMMY";
  }
  TRGDummy2DB::~TRGDummy2DB(){}
}//ns lumi
#include "RecoLuminosity/LumiProducer/interface/DataPipeFactory.h"
DEFINE_EDM_PLUGIN(lumi::DataPipeFactory,lumi::TRGDummy2DB,"TRGDummy2DB");
#endif
