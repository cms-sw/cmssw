#ifndef RecoLuminosity_LumiProducer_LumiDummy2DB_H 
#define RecoLuminosity_LumiProducer_LumiDummy2DB_H 
#include "RelationalAccess/ConnectionService.h"
#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/ITransaction.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/ITableDataEditor.h"
#include "RelationalAccess/IBulkOperation.h"
#include "RelationalAccess/ITypeConverter.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/AttributeSpecification.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/Blob.h"
#include "CoralBase/Exception.h"

#include "RecoLuminosity/LumiProducer/interface/DataPipe.h"
#include "RecoLuminosity/LumiProducer/interface/ConstantDef.h"
#include "RecoLuminosity/LumiProducer/interface/LumiNames.h"
#include "RecoLuminosity/LumiProducer/interface/idDealer.h"
#include "RecoLuminosity/LumiProducer/interface/Exception.h"
#include "RecoLuminosity/LumiProducer/interface/DBConfig.h"
//#include <iostream>
#include <cstring>
namespace lumi{
  class LumiDummy2DB : public DataPipe{
  public:
    LumiDummy2DB(const std::string& dest);
    virtual unsigned long long retrieveData( unsigned int );
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
  unsigned long long LumiDummy2DB::retrieveData( unsigned int runnum){
    //
    //generate dummy data for lumi summary and detail for the given run and write data to LumiDB
    //
    coral::ConnectionService* svc=new coral::ConnectionService;
    lumi::DBConfig dbconf(*svc);
    if(!m_authpath.empty()){
      dbconf.setAuthentication(m_authpath);
    }
    coral::ISessionProxy* session=svc->connect(m_dest,coral::Update);
    coral::ITypeConverter& tpc=session->typeConverter();
    tpc.setCppTypeForSqlType("unsigned int","NUMBER(10)");
    try{
      unsigned int totallumils=35;
      unsigned int totalcmsls=32;
      session->transaction().start(false);
      coral::ISchema& schema=session->nominalSchema();
      lumi::idDealer idg(schema);
      coral::ITable& summarytable=schema.tableHandle(LumiNames::lumisummaryTableName());
      coral::ITable& detailtable=schema.tableHandle(LumiNames::lumidetailTableName());
      coral::AttributeList summaryData;
      summaryData.extend("LUMISUMMARY_ID",typeid(unsigned long long));
      summaryData.extend("RUNNUM",typeid(unsigned int));
      summaryData.extend("CMSLSNUM",typeid(unsigned int));
      summaryData.extend("LUMILSNUM",typeid(unsigned int));
      summaryData.extend("LUMIVERSION",typeid(std::string));
      summaryData.extend("DTNORM",typeid(float));
      summaryData.extend("LHCNORM",typeid(float));
      summaryData.extend("INSTLUMI",typeid(float));
      summaryData.extend("INSTLUMIERROR",typeid(float));
      summaryData.extend("INSTLUMIQUALITY",typeid(short));
      summaryData.extend("LUMISECTIONQUALITY",typeid(short));
      summaryData.extend("CMSALIVE",typeid(short));
      summaryData.extend("STARTORBIT",typeid(unsigned int));
      summaryData.extend("NUMORBIT",typeid(unsigned int));
      summaryData.extend("BEAMENERGY",typeid(float));
      summaryData.extend("BEAMSTATUS",typeid(std::string));

      coral::IBulkOperation* summaryInserter=summarytable.dataEditor().bulkInsert(summaryData,totallumils);
      coral::AttributeList detailData;
      detailData.extend("LUMIDETAIL_ID",typeid(unsigned long long));
      detailData.extend("LUMISUMMARY_ID",typeid(unsigned long long));
      detailData.extend("BXLUMIVALUE",typeid(coral::Blob));
      detailData.extend("BXLUMIERROR",typeid(coral::Blob));
      detailData.extend("BXLUMIQUALITY",typeid(coral::Blob));
      detailData.extend("ALGONAME",typeid(std::string));
      coral::IBulkOperation* detailInserter=detailtable.dataEditor().bulkInsert(detailData,totallumils*N_LUMIALGO);
      //loop over lumi LS
      unsigned long long& lumisummary_id=summaryData["LUMISUMMARY_ID"].data<unsigned long long>();
      unsigned int& lumirunnum = summaryData["RUNNUM"].data<unsigned int>();
      unsigned int& cmslsnum=summaryData["CMSLSNUM"].data<unsigned int>();
      unsigned int& lumilsnum=summaryData["LUMILSNUM"].data<unsigned int>();
      std::string& lumiversion = summaryData["LUMIVERSION"].data<std::string>();
      float& dtnorm = summaryData["DTNORM"].data<float>();
      float& lhcnorm = summaryData["LHCNORM"].data<float>();
      float& instlumi = summaryData["INSTLUMI"].data<float>();
      float& instlumierror = summaryData["INSTLUMIERROR"].data<float>();
      short& instlumiquality = summaryData["INSTLUMIQUALITY"].data<short>();
      short& lumisectionquality = summaryData["LUMISECTIONQUALITY"].data<short>();
      short& cmsalive = summaryData["CMSALIVE"].data<short>();
      unsigned int& startorbit=summaryData["STARTORBIT"].data<unsigned int>();
      unsigned int& numorbit= summaryData["NUMORBIT"].data<unsigned int>();
      float& beamenergy= summaryData["BEAMENERGY"].data<float>();
      std::string& beamstatus= summaryData["BEAMSTATUS"].data<std::string>();

      unsigned long long& lumidetail_id=detailData["LUMIDETAIL_ID"].data<unsigned long long>();
      unsigned long long& d2lumisummary_id=detailData["LUMISUMMARY_ID"].data<unsigned long long>();
      coral::Blob& bxlumivalue=detailData["BXLUMIVALUE"].data<coral::Blob>();
      coral::Blob& bxlumierror=detailData["BXLUMIERROR"].data<coral::Blob>();
      coral::Blob& bxlumiquality=detailData["BXLUMIQUALITY"].data<coral::Blob>();
      std::string& algoname=detailData["ALGONAME"].data<std::string>();
      for(unsigned int i=1;i<=totallumils;++i){
	lumisummary_id = idg.generateNextIDForTable(LumiNames::lumisummaryTableName());
	lumilsnum=i;
	lumirunnum = runnum;
	lumiversion = "0";
	dtnorm = 1.05;
	lhcnorm = 1.2;
	instlumi = 0.9;
	instlumierror = 0.01;
	instlumiquality = 8;
	lumisectionquality = 16;
	//	unsigned int cmslsnum = 0;
	short iscmsalive = 0;
	if(i<=totalcmsls){
	  iscmsalive=1;
	  cmslsnum=i;
	}
	cmsalive=iscmsalive;
	startorbit=2837495;
	numorbit=34566;
	beamenergy=362;
	beamstatus="stable";
	//fetch a new id value 
	//insert the new row
	summaryInserter->processNextIteration();
	summaryInserter->flush();
	d2lumisummary_id=i;
	for( unsigned int j=0; j<N_LUMIALGO; ++j ){
	  lumidetail_id=idg.generateNextIDForTable(LumiNames::lumidetailTableName());
	  if(j==0) algoname=std::string("ET");
	  if(j==1) algoname=std::string("OCC1");
	  if(j==2) algoname=std::string("OCC2");
	  float lumivalue[N_BX];
	  std::memset((void*)&lumivalue,0,sizeof(float)*N_BX);
	  float lumierror[N_BX];
	  std::memset((void*)&lumierror,0,sizeof(float)*N_BX );
	  short lumiquality[N_BX];
	  std::memset((void*)&lumiquality,0,sizeof(short)*N_BX );
	  bxlumivalue.resize(sizeof(float)*N_BX);
	  bxlumierror.resize(sizeof(float)*N_BX);
	  bxlumiquality.resize(sizeof(short)*N_BX);
	  void* bxlumivalueStartAddress=bxlumivalue.startingAddress();
	  void* bxlumierrorStartAddress=bxlumierror.startingAddress();
	  void* bxlumiqualityStartAddress=bxlumiquality.startingAddress();
	  for( unsigned int k=0; k<N_BX; ++k ){	    
	    lumivalue[k]=1.5;
	    lumierror[k]=0.1;
	    lumiquality[k]=1;
	  }
	  std::memmove(bxlumivalueStartAddress,lumivalue,sizeof(float)*N_BX);
	  std::memmove(bxlumierrorStartAddress,lumierror,sizeof(float)*N_BX);
	  std::memmove(bxlumiqualityStartAddress,lumiquality,sizeof(short)*N_BX);
	  detailInserter->processNextIteration();
	}
      }
      detailInserter->flush();
      delete summaryInserter;
      delete detailInserter;
    }catch( const coral::Exception& er){
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
