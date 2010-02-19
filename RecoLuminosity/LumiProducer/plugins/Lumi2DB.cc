#ifndef RecoLuminosity_LumiProducer_Lumi2DB_H 
#define RecoLuminosity_LumiProducer_Lumi2DB_H
#include "RelationalAccess/IAuthenticationService.h"
#include "RelationalAccess/IConnectionService.h"
#include "RelationalAccess/IConnectionServiceConfiguration.h"
#include "CoralKernel/Context.h"
#include "CoralKernel/IHandle.h"
#include "CoralKernel/IProperty.h"
#include "CoralKernel/IPropertyManager.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/AttributeSpecification.h"
#include "CoralBase/Blob.h"
#include "CoralBase/Exception.h"
#include "CoralBase/TimeStamp.h"
#include "CoralBase/MessageStream.h"
#include "RelationalAccess/AccessMode.h"
#include "RelationalAccess/ConnectionService.h"
#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/IConnectionServiceConfiguration.h"
#include "RelationalAccess/ITransaction.h"
#include "RelationalAccess/ITypeConverter.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ICursor.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/IView.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/ITableDataEditor.h"
#include "RelationalAccess/IBulkOperation.h"
#include "RecoLuminosity/LumiProducer/interface/LumiRawDataStructures.h"
#include "RecoLuminosity/LumiProducer/interface/DataPipe.h"
#include "RecoLuminosity/LumiProducer/interface/LumiNames.h"
#include "RecoLuminosity/LumiProducer/interface/idDealer.h"
#include <iostream>
#include <cstring>
#include "TFile.h"
#include "TTree.h"
namespace lumi{
  class Lumi2DB : public DataPipe{
  public:
    Lumi2DB(const std::string& dest);
    virtual void retrieveData( unsigned int );
    virtual const std::string dataType() const;
    virtual const std::string sourceType() const;
    virtual ~Lumi2DB();
    
    struct PerBXData{
      int idx;
      float lumivalue;
      float lumierr;
      int lumiquality;
    };
    
    struct PerLumiData{
      unsigned int cmslsnr;
      unsigned int lumilsnr;
      unsigned int startorbit;
      float lumiavg;
      std::vector<PerBXData> bxET;
      std::vector<PerBXData> bxOCC1;
      std::vector<PerBXData> bxOCC2;
    };
    typedef std::vector<PerLumiData> LumiResult;
    
  };//cl Lumi2DB
}//ns lumi

//
//implementation
//
lumi::Lumi2DB::Lumi2DB(const std::string& dest):DataPipe(dest){}

void lumi::Lumi2DB::retrieveData( unsigned int runnumber){
  lumi::Lumi2DB::LumiResult lumiresult;
  TFile* source=TFile::Open(m_source.c_str(),"READ");
  TTree *hlxtree = (TTree*)source->Get("HLXData");
  if(!hlxtree){
    throw std::runtime_error(std::string("non-existing HLXData "));
  }
  //hlxtree->Print();
  std::auto_ptr<HCAL_HLX::LUMI_SECTION> localSection(new HCAL_HLX::LUMI_SECTION);
  HCAL_HLX::LUMI_SECTION_HEADER* lumiheader = &(localSection->hdr);
  HCAL_HLX::LUMI_SUMMARY* lumisummary = &(localSection->lumiSummary);
  HCAL_HLX::LUMI_DETAIL* lumidetail = &(localSection->lumiDetail);
  
  hlxtree->SetBranchAddress("Header.",&lumiheader);
  hlxtree->SetBranchAddress("Summary.",&lumisummary);
  hlxtree->SetBranchAddress("Detail.",&lumidetail);
  
  size_t nentries=hlxtree->GetEntries();
  size_t ncmslumi=0;
  //std::cout<<"processing total lumi lumisection "<<nentries<<std::endl;
  //size_t lumisecid=0;
  //unsigned int lumilumisecid=0;
  //runnumber=lumiheader->runNumber;
  for(size_t i=0;i<nentries;++i){
    hlxtree->GetEntry(i);
    if(!lumiheader->bCMSLive){
      std::cout<<"non-CMS LS "<<lumiheader->sectionNumber<<std::endl;
      continue;
    }else{
      ++ncmslumi;
    }
    lumi::Lumi2DB::PerLumiData h;
    h.bxET.reserve(3564);
    h.bxOCC1.reserve(3564);
    h.bxOCC2.reserve(3564);

    //runnumber=lumiheader->runNumber;
    //if(runnumber!=m_run) throw std::runtime_error(std::string("requested run ")+this->int2str(m_run)+" does not match runnumber in the data header "+this->int2str(runnumber));
    h.lumilsnr=lumiheader->sectionNumber;
    h.cmslsnr=ncmslumi;//we record cms lumils
    h.startorbit=lumiheader->startOrbit;
    h.lumiavg=lumisummary->InstantLumi;
    
    for(size_t i=0;i<3564;++i){
      lumi::Lumi2DB::PerBXData bET;
      lumi::Lumi2DB::PerBXData bOCC1;
      lumi::Lumi2DB::PerBXData bOCC2;
      bET.idx=i+1;
      bET.lumivalue=lumidetail->ETLumi[i];
      bET.lumierr=lumidetail->ETLumiErr[i];
      bET.lumiquality=lumidetail->ETLumiQlty[i];      
      //bxinfoET.push_back(lumi::BunchCrossingInfo(i+1,lumidetail->ETLumi[i],lumidetail->ETLumiErr[i],lumidetail->ETLumiQlty[i]));
      h.bxET.push_back(bET);

      bOCC1.idx=i+1;
      bOCC1.lumivalue=lumidetail->OccLumi[0][i];
      bOCC1.lumierr=lumidetail->OccLumiErr[0][i];
      bOCC1.lumiquality=lumidetail->OccLumiQlty[0][i]; 
      h.bxOCC1.push_back(bOCC1);
      //bxinfoOCC1.push_back(lumi::BunchCrossingInfo(i+1,lumidetail->OccLumi[0][i],lumidetail->OccLumiErr[0][i],lumidetail->OccLumiQlty[0][i]));
      
      bOCC2.idx=i+1;
      bOCC2.lumivalue=lumidetail->OccLumi[1][i];
      bOCC2.lumierr=lumidetail->OccLumiErr[1][i];
      bOCC2.lumiquality=lumidetail->OccLumiQlty[1][i]; 
      h.bxOCC2.push_back(bOCC2);
      //bxinfoOCC2.push_back(lumi::BunchCrossingInfo(i+1,lumidetail->OccLumi[1][i],lumidetail->OccLumiErr[1][i],lumidetail->OccLumiQlty[1][i]));
      
    }
    lumiresult.push_back(h);
  }
  coral::ConnectionService* svc=new coral::ConnectionService;
  coral::ISessionProxy* session=svc->connect(m_dest,coral::Update);
  try{
    unsigned int totallumils=35;
    unsigned int totalcmsls=32;
    session->transaction().start(false);
    coral::ISchema& schema=session->nominalSchema();
    lumi::idDealer idg(schema);
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
    detailData.extend<coral::Blob>("BXLUMIVALUE");
    detailData.extend<coral::Blob>("BXLUMIERROR");
    detailData.extend<coral::Blob>("BXLUMIQUALITY");
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
    coral::Blob& bxlumivalue=detailData["BXLUMIVALUE"].data<coral::Blob>();
    coral::Blob& bxlumierror=detailData["BXLUMIERROR"].data<coral::Blob>();
    coral::Blob& bxlumiquality=detailData["BXLUMIQUALITY"].data<coral::Blob>();
    std::string& algoname=detailData["ALGONAME"].data<std::string>();
    for(unsigned int i=1;i<=totallumils;++i){
      lumisummary_id = idg.generateNextIDForTable(LumiNames::lumisummaryTableName());
      lumirunnum = runnumber;
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
	lumidetail_id=idg.generateNextIDForTable(LumiNames::lumidetailTableName());
	if(j==0) algoname=std::string("ET");
	if(j==1) algoname=std::string("OCC1");
	if(j==2) algoname=std::string("OCC2");
	float lumivalue[3564];
	std::memset((void*)&lumivalue,0,sizeof(float)*3564 );
	float lumierror[3564];
	std::memset((void*)&lumierror,0,sizeof(float)*3564 );
	int lumiquality[3564];
	std::memset((void*)&lumiquality,0,sizeof(int)*3564 );
	bxlumivalue.resize(sizeof(float)*3564);
	bxlumierror.resize(sizeof(float)*3564);
	bxlumiquality.resize(sizeof(int)*3564);
	void* bxlumivalueStartAddress=bxlumivalue.startingAddress();
	void* bxlumierrorStartAddress=bxlumierror.startingAddress();
	void* bxlumiqualityStartAddress=bxlumiquality.startingAddress();
	for( unsigned int k=0; k<3546; ++k ){	    
	  lumivalue[k]=1.5;
	  lumierror[k]=0.1;
	  lumiquality[k]=1;
	}
	std::memmove(bxlumivalueStartAddress,lumivalue,sizeof(float)*3564);
	std::memmove(bxlumierrorStartAddress,lumierror,sizeof(float)*3564);
	std::memmove(bxlumiqualityStartAddress,lumiquality,sizeof(int)*3564);
	detailInserter->processNextIteration();
      }
    }
    detailInserter->flush();
    summaryInserter->flush();
    delete summaryInserter;
    delete detailInserter;
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
const std::string lumi::Lumi2DB::dataType() const{
  return "LUMI";
}
const std::string lumi::Lumi2DB::sourceType() const{
  return "DB";
}
lumi::Lumi2DB::~Lumi2DB(){}

#include "RecoLuminosity/LumiProducer/interface/DataPipeFactory.h"
DEFINE_EDM_PLUGIN(lumi::DataPipeFactory,lumi::Lumi2DB,"Lumi2DB");
#endif
