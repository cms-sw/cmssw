#ifndef RecoLuminosity_LumiProducer_Lumi2DB_H 
#define RecoLuminosity_LumiProducer_Lumi2DB_H
#include "RelationalAccess/IAuthenticationService.h"
#include "RelationalAccess/IConnectionService.h"
#include "RelationalAccess/IConnectionServiceConfiguration.h"
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
#include "RecoLuminosity/LumiProducer/interface/Exception.h"
#include <iostream>
#include <cstring>
#include <cstdlib>
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
    struct LumiSource{
      unsigned int run;
      unsigned int firstsection;
      char version[8];
      char datestr[9];
    };
    struct PerBXData{
      //int idx;
      float lumivalue;
      float lumierr;
      short lumiquality;
    };
    struct PerLumiData{
      float dtnorm;
      float lhcnorm;
      float instlumi;
      float instlumierror;
      short instlumiquality;
      short lumisectionquality;
      bool cmsalive;
      unsigned int cmslsnr;
      unsigned int lumilsnr;
      unsigned int startorbit;

      std::vector<PerBXData> bxET;
      std::vector<PerBXData> bxOCC1;
      std::vector<PerBXData> bxOCC2;
    };
    typedef std::vector<PerLumiData> LumiResult;
  private:
    void parseSourceString(lumi::Lumi2DB::LumiSource& result)const;
  };//cl Lumi2DB
}//ns lumi

//
//implementation
//
lumi::Lumi2DB::Lumi2DB(const std::string& dest):DataPipe(dest){}

void 
lumi::Lumi2DB::parseSourceString(lumi::Lumi2DB::LumiSource& result)const{
  //parse lumi source file name
  if(m_source.length()==0) throw lumi::Exception("lumi source is not set","parseSourceString","Lumi2DB");
  size_t tempIndex = m_source.rfind(".");
  size_t nameLength = m_source.length();
  std::string subFileName = m_source.substr(0,nameLength - tempIndex);
  
  std::vector<std::string> fileNamePieces;
  std::string::size_type lastPos=m_source.find_first_not_of("_",0);
  std::string::size_type pos = m_source.find_first_of("_",lastPos);
  std::vector<std::string> pieces;
  while( std::string::npos != pos || std::string::npos != lastPos){
    pieces.push_back(m_source.substr(lastPos,pos-lastPos));
    lastPos=m_source.find_first_not_of("_",pos);
    pos=m_source.find_first_of("_",lastPos);
  }
  if( pieces[0]!="CMS" || pieces[1]!="LUMI" ||  pieces[2]!="RAW" ){
    throw lumi::Exception("not lumi raw data file CMS_LUMI_RAW","parseSourceString","Lumi2DB");
  }

  std::strcpy(result.datestr,fileNamePieces[3].c_str());
  std::strcpy(result.version,fileNamePieces[6].c_str());
  result.run = atoi(pieces[4].c_str());
  result.firstsection = atoi(pieces[5].c_str());
}

void 
lumi::Lumi2DB::retrieveData( unsigned int runnumber){
  lumi::Lumi2DB::LumiResult lumiresult;
  //check filename is in  lumiraw format
  lumi::Lumi2DB::LumiSource filenamecontent;
  try{
    parseSourceString(filenamecontent);
  }catch(const lumi::Exception& er){
    std::cout<<er.what()<<std::endl;
    throw er;
  }
  if( filenamecontent.run!=runnumber ){
    throw lumi::Exception("runnumber in file name does not match requested run number","retrieveData","Lumi2DB");
  }
  TFile* source=TFile::Open(m_source.c_str(),"READ");
  TTree *hlxtree = (TTree*)source->Get("HLXData");
  if(!hlxtree){
    throw lumi::Exception(std::string("non-existing HLXData "),"retrieveData","Lumi2DB");
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
  std::cout<<"processing total lumi lumisection "<<nentries<<std::endl;
  //size_t lumisecid=0;
  //unsigned int lumilumisecid=0;
  //runnumber=lumiheader->runNumber;
  for(size_t i=0;i<nentries;++i){
    lumi::Lumi2DB::PerLumiData h;
    h.cmsalive=true;
    hlxtree->GetEntry(i);
    if(!lumiheader->bCMSLive){
      std::cout<<"non-CMS LS "<<lumiheader->sectionNumber<<std::endl;
      h.cmsalive=false;
      continue;
    }else{
      ++ncmslumi;
    }
    h.bxET.reserve(3564);
    h.bxOCC1.reserve(3564);
    h.bxOCC2.reserve(3564);
    
    //runnumber=lumiheader->runNumber;
    //if(runnumber!=m_run) throw std::runtime_error(std::string("requested run ")+this->int2str(m_run)+" does not match runnumber in the data header "+this->int2str(runnumber));
    h.lumilsnr=lumiheader->sectionNumber;
    h.startorbit=lumiheader->startOrbit;
    h.cmslsnr=ncmslumi;//we record cms lumils
    h.instlumi=lumisummary->InstantLumi;
    h.instlumierror=lumisummary->InstantLumiErr;
    h.lumisectionquality=lumisummary->InstantLumiQlty;
    h.dtnorm=lumisummary->DeadTimeNormalization;
    h.lhcnorm=lumisummary->LHCNormalization;
    for(size_t i=0;i<3564;++i){
      lumi::Lumi2DB::PerBXData bET;
      lumi::Lumi2DB::PerBXData bOCC1;
      lumi::Lumi2DB::PerBXData bOCC2;
      //bET.idx=i+1;
      bET.lumivalue=lumidetail->ETLumi[i];
      bET.lumierr=lumidetail->ETLumiErr[i];
      bET.lumiquality=lumidetail->ETLumiQlty[i];      
      h.bxET.push_back(bET);

      //bOCC1.idx=i+1;
      bOCC1.lumivalue=lumidetail->OccLumi[0][i];
      bOCC1.lumierr=lumidetail->OccLumiErr[0][i];
      bOCC1.lumiquality=lumidetail->OccLumiQlty[0][i]; 
      h.bxOCC1.push_back(bOCC1);
          
      //bOCC2.idx=i+1;
      bOCC2.lumivalue=lumidetail->OccLumi[1][i];
      bOCC2.lumierr=lumidetail->OccLumiErr[1][i];
      bOCC2.lumiquality=lumidetail->OccLumiQlty[1][i]; 
      h.bxOCC2.push_back(bOCC2);
    }
    lumiresult.push_back(h);
  }
  coral::ConnectionService* svc=new coral::ConnectionService;
  coral::ISessionProxy* session=svc->connect(m_dest,coral::Update);
  unsigned int totallumils=lumiresult.size();
  try{
    session->transaction().start(false);
    coral::ISchema& schema=session->nominalSchema();
    lumi::idDealer idg(schema);
    coral::ITable& summarytable=schema.tableHandle(LumiNames::lumisummaryTableName());
    coral::ITable& detailtable=schema.tableHandle(LumiNames::lumidetailTableName());
    coral::AttributeList summaryData;
    summaryData.extend<unsigned long long>("LUMISUMMARY_ID");
    summaryData.extend<unsigned int>("RUNNUM");
    summaryData.extend<unsigned int>("CMSLSNUM");
    summaryData.extend<unsigned int>("LUMILSNUM");
    summaryData.extend<std::string>("LUMIVERSION");
    summaryData.extend<float>("DTNORM");
    summaryData.extend<float>("LHCNORM");
    summaryData.extend<float>("INSTLUMI");
    summaryData.extend<float>("INSTLUMIERROR");
    summaryData.extend<short>("INSTLUMIQUALITY");
    summaryData.extend<short>("LUMISECTIONQUALITY");
    summaryData.extend<bool>("CMSALIVE");
    summaryData.extend<unsigned int>("STARTORBIT");
    coral::IBulkOperation* summaryInserter=summarytable.dataEditor().bulkInsert(summaryData,totallumils);
    
    coral::AttributeList detailData;
    detailData.extend<unsigned long long>("LUMIDETAIL_ID");
    detailData.extend<unsigned long long>("LUMISUMMARY_ID");
    detailData.extend<coral::Blob>("BXLUMIVALUE");
    detailData.extend<coral::Blob>("BXLUMIERROR");
    detailData.extend<coral::Blob>("BXLUMIQUALITY");
    detailData.extend<std::string>("ALGONAME");
    coral::IBulkOperation* detailInserter=detailtable.dataEditor().bulkInsert(detailData,totallumils*3);
    //loop over lumi LS
    unsigned long long& lumisummary_id=summaryData["LUMISUMMARY_ID"].data<unsigned long long>();
    unsigned int& lumirunnum = summaryData["RUNNUM"].data<unsigned int>();
    std::string& lumiversion = summaryData["LUMIVERSION"].data<std::string>();
    float& dtnorm = summaryData["DTNORM"].data<float>();
    float& lhcnorm = summaryData["LHCNORM"].data<float>();
    float& instlumi = summaryData["INSTLUMI"].data<float>();
    float& instlumierror = summaryData["INSTLUMIERROR"].data<float>();
    short& instlumiquality = summaryData["INSTLUMIQUALITY"].data<short>();
    short& lumisectionquality = summaryData["LUMISECTIONQUALITY"].data<short>();
    bool& cmsalive = summaryData["CMSALIVE"].data<bool>();
    unsigned int& lumilsnr = summaryData["LUMILSNUM"].data<unsigned int>();
    unsigned int& cmslsnr = summaryData["CMSLSNUM"].data<unsigned int>();
    unsigned int& startorbit = summaryData["STARTORBIT"].data<unsigned int>();

    unsigned long long& lumidetail_id=detailData["LUMIDETAIL_ID"].data<unsigned long long>();
    unsigned long long& d2lumisummary_id=detailData["LUMISUMMARY_ID"].data<unsigned long long>();
    coral::Blob& bxlumivalue=detailData["BXLUMIVALUE"].data<coral::Blob>();
    coral::Blob& bxlumierror=detailData["BXLUMIERROR"].data<coral::Blob>();
    coral::Blob& bxlumiquality=detailData["BXLUMIQUALITY"].data<coral::Blob>();
    std::string& algoname=detailData["ALGONAME"].data<std::string>();


    
    lumi::Lumi2DB::LumiResult::const_iterator lumiIt;
    lumi::Lumi2DB::LumiResult::const_iterator lumiBeg=lumiresult.begin();
    lumi::Lumi2DB::LumiResult::const_iterator lumiEnd=lumiresult.end();
    for(lumiIt=lumiBeg;lumiIt!=lumiEnd;++lumiIt){
      lumisummary_id = idg.generateNextIDForTable(LumiNames::lumisummaryTableName());      
      lumirunnum = runnumber;
      lumiversion = std::string(filenamecontent.version);
      dtnorm = lumiIt->dtnorm;
      lhcnorm = lumiIt->lhcnorm;
      instlumi = lumiIt->instlumi;
      instlumierror = lumiIt->instlumierror;
      instlumiquality = lumiIt->instlumiquality;
      lumisectionquality = lumiIt->lumisectionquality;
      cmsalive = lumiIt->cmsalive;
      cmslsnr = lumiIt->cmslsnr;
      lumilsnr = lumiIt->lumilsnr;
      startorbit = lumiIt->startorbit;

      //fetch a new id value 
      //insert the new row
      summaryInserter->processNextIteration();
      for( unsigned int j=0; j<3; ++j ){
	lumidetail_id=idg.generateNextIDForTable(LumiNames::lumidetailTableName());
	d2lumisummary_id=lumisummary_id;
	std::vector<PerBXData>::const_iterator bxIt;
	std::vector<PerBXData>::const_iterator bxBeg;
	std::vector<PerBXData>::const_iterator bxEnd;
	if(j==0) {
	  algoname=std::string("ET");
	  bxBeg=lumiIt->bxET.begin();
	  bxEnd=lumiIt->bxET.end();
	}
	if(j==1) {
	  algoname=std::string("OCC1");
	  bxBeg=lumiIt->bxOCC1.begin();
	  bxEnd=lumiIt->bxOCC1.end();
	}
	if(j==2) {
	  algoname=std::string("OCC2");
	  bxBeg=lumiIt->bxOCC2.begin();
	  bxEnd=lumiIt->bxOCC2.end();
	}
	float lumivalue[3564];
	std::memset((void*)&lumivalue,0,sizeof(float)*3564 );
	float lumierror[3564];
	std::memset((void*)&lumierror,0,sizeof(float)*3564 );
	int lumiquality[3564];
	std::memset((void*)&lumiquality,0,sizeof(short)*3564 );
	bxlumivalue.resize(sizeof(float)*3564);
	bxlumierror.resize(sizeof(float)*3564);
	bxlumiquality.resize(sizeof(short)*3564);
	void* bxlumivalueStartAddress=bxlumivalue.startingAddress();
	void* bxlumierrorStartAddress=bxlumierror.startingAddress();
	void* bxlumiqualityStartAddress=bxlumiquality.startingAddress();
	unsigned int k=0;
	for( bxIt=bxBeg;bxIt!=bxEnd;++bxIt,++k  ){	    
	  lumivalue[k]=bxIt->lumivalue;
	  lumierror[k]=bxIt->lumierr;
	  lumiquality[k]=bxIt->lumiquality;
	}
	std::memmove(bxlumivalueStartAddress,lumivalue,sizeof(float)*3564);
	std::memmove(bxlumierrorStartAddress,lumierror,sizeof(float)*3564);
	std::memmove(bxlumiqualityStartAddress,lumiquality,sizeof(short)*3564);
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
