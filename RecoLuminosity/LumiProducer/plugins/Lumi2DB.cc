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
#include "RecoLuminosity/LumiProducer/interface/DataPipe.h"
#include "RecoLuminosity/LumiProducer/interface/LumiRawDataStructures.h"
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
  const std::string lumirawname("test.root");
  TFile* source=TFile::Open(lumirawname.c_str(),"READ");
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
