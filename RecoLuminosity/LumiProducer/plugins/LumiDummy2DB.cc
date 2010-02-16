#ifndef RecoLuminosity_LumiProducer_LumiDummy2DB_H 
#define RecoLuminosity_LumiProducer_LumiDummy2DB_H 
#include "RelationalAccess/ConnectionService.h"
#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/ITransaction.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/ITableDataEditor.h"
#include "RelationalAccess/IBulkOperation.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ICursor.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/AttributeSpecification.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/Blob.h"
#include "CoralBase/Exception.h"

#include "RecoLuminosity/LumiProducer/interface/DataPipe.h"
#include "RecoLuminosity/LumiProducer/interface/LumiNames.h"
#include <iostream>
namespace lumi{
  class LumiDummy2DB : public DataPipe{
  public:
    LumiDummy2DB(const std::string& dest);
    virtual void retrieveRun( unsigned int );
    virtual const std::string dataType() const;
    virtual const std::string sourceType() const;
    virtual ~LumiDummy2DB();
  };//cl LumiDummy2DB
  //
  //implementation
  //
  LumiDummy2DB::LumiDummy2DB( const std::string& dest):DataPipe(dest){
    //check the format of dest
  }
  void LumiDummy2DB::retrieveRun( unsigned int runnum){
    //
    //generate dummy data for lumi summary and detail for the given run and write data to LumiDB
    //
    coral::ConnectionService* svc=new coral::ConnectionService;
    coral::ISessionProxy* session=svc->connect(m_dest,coral::Update);
    try{
      unsigned int totallumils=35;
      unsigned int totalcmsls=32;
      session->transaction().start(false);
      coral::ISchema& schema=session->nominalSchema();
      coral::ITable& summarytable=schema.tableHandle(LumiNames::lumisummaryTableName());
      coral::ITable& detailtable=schema.tableHandle(LumiNames::lumidetailTableName());
      coral::AttributeList summaryData;
      summaryData.extend<unsigned long long>("LUMISUMMARY_ID");
      summaryData.extend<unsigned int>("RUNNUM");
      summaryData.extend<std::string>("LUMIVERSION");
      summaryData.extend<float>("DTNORM");
      summaryData.extend<float>("LUMINORM");
      summaryData.extend<float>("INSTLUMI");
      summaryData.extend<float>("INSTLUMIERROR");
      summaryData.extend<short>("INSTLUMIQUALITY");
      summaryData.extend<short>("LUMISECTIONQUALITY");
      summaryData.extend<bool>("CMSALIVE");
      summaryData.extend<unsigned int>("LUMILSNUM");
      coral::IBulkOperation* summaryInserter=summarytable.dataEditor().bulkInsert(summaryData,totallumils);

      coral::AttributeList detailData;
      detailData.extend<unsigned long long>("LUMIDETAIL_ID");
      detailData.extend<unsigned long long>("LUMISUMMARY_ID");
      detailData.extend<coral::Blob>("BXINFO");
      detailData.extend<std::string>("ALGONAME");

      coral::IBulkOperation* detailInserter=detailtable.dataEditor().bulkInsert(detailData,totallumils);
      //loop over lumi LS
      unsigned long long& lumisummary_id=summaryData["LUMISUMMARY_ID"].data<unsigned long long>();
      unsigned int& lumirunnum = summaryData["RUNNUM"].data<unsigned int>();
      std::string& lumiversion = summaryData["LUMIVERSION"].data<std::string>();
      float& dtnorm = summaryData["DTNORM"].data<float>();
      float& luminorm = summaryData["LUMINORM"].data<float>();
      float& instlumi = summaryData["INSTLUMI"].data<float>();
      float& instlumierror = summaryData["INSTLUMIERROR"].data<float>();
      short& instlumiquality = summaryData["INSTLUMIQUALITY"].data<short>();
      short& lumisectionquality = summaryData["LUMISECTIONQUALITY"].data<short>();
      bool& cmsalive = summaryData["CMSALIVE"].data<bool>();
      unsigned int& lumilsnum = summaryData["LUMILSNUM"].data<unsigned int>();

      unsigned long long& lumidetail_id=detailData["LUMIDETAIL_ID"].data<unsigned long long>();
      unsigned long long& d2lumisummary_id=detailData["LUMISUMMARY_ID"].data<unsigned long long>();
      //coral::Blob& bxinfo=detailData["BXINFO"].data<coral::Blob>();
      std::string& algoname=detailData["ALGONAME"].data<std::string>();
      
      for(unsigned int i=1;i<=totallumils;++i){
	lumisummary_id = i;
	lumirunnum = runnum;
	lumiversion = "0";
	dtnorm = 1.05;
	luminorm = 1.2;
	instlumi = 0.9;
	instlumierror = 0.01;
	instlumiquality = 8;
	lumisectionquality = 16;
	unsigned int cmslsnum = 0;
	bool iscmsalive = false;
	if(i<=totalcmsls){
	  iscmsalive=true;
	  cmslsnum=i;
	}
	cmsalive=iscmsalive;
	lumilsnum=cmslsnum;
	//fetch a new id value 
	//insert the new row
	summaryInserter->processNextIteration();
	d2lumisummary_id=i;
	for( unsigned int j=0; j<3; ++j ){
	  lumidetail_id=i*100+j;
	  if(j==0) algoname=std::string("ET");
	  if(j==1) algoname=std::string("OCC1");
	  if(j==2) algoname=std::string("OCC2");
	  //for( unsigned int k=0; k<3546; ++k ){	    
	  //}
	  detailInserter->processNextIteration();
	}
      }
      summaryInserter->flush();
      delete summaryInserter;
      detailInserter->flush();
      delete detailInserter;
    }catch( const coral::Exception& er){
      std::cout<<"database problem "<<er.what()<<std::endl;
      session->transaction().rollback();
      delete session;
      delete svc;
    }
    //delete detailInserter;
    session->transaction().commit();
    delete session;
    delete svc;
  }
  const std::string LumiDummy2DB::dataType() const{
    return "LUMI";
  }
  const std::string LumiDummy2DB::sourceType() const{
    return "DUMMY";
  }
  LumiDummy2DB::~LumiDummy2DB(){}
}//ns lumi
#include "RecoLuminosity/LumiProducer/interface/DataPipeFactory.h"
DEFINE_EDM_PLUGIN(lumi::DataPipeFactory,lumi::LumiDummy2DB,"LumiDummy2DB");
#endif
