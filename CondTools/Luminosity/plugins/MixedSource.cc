#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "CondCore/DBCommon/interface/Time.h"
#include "CondFormats/Luminosity/interface/LumiSectionData.h"
#include "CondTools/Luminosity/interface/LumiRetrieverFactory.h"
#include "LumiDataStructures.h"
#include "MixedSource.h"
#include <memory>
#include <iostream>
#include "TFile.h"
#include "TTree.h"

lumi::MixedSource::MixedSource(const edm::ParameterSet& pset):LumiRetrieverBase(pset){
  m_filename=pset.getParameter<std::string>("lumiFileName");
  m_source=TFile::Open(m_filename.c_str(),"READ");
  //m_source->GetListOfKeys()->Print();
  std::string::size_type idx,pos;
  idx=m_filename.rfind("_");
  pos=m_filename.rfind(".");
  if( idx == std::string::npos ){
  }
  m_lumiversion=m_filename.substr(idx+1,pos-idx-1);
}

const std::string
lumi::MixedSource::fill(std::vector< std::pair<lumi::LumiSectionData*,cond::Time_t> >& result , bool allowForceFirstSince ){
  m_source->ls();
  //TTree *runsummary = (TTree*)m_source->Get("RunSummary");
  TTree *hlxtree = (TTree*)m_source->Get("HLXData");
  unsigned int runnumber=0;
  if(hlxtree){
    //runsummary->Print();
    hlxtree->Print();
    std::auto_ptr<HCAL_HLX::LUMI_SECTION> localSection(new HCAL_HLX::LUMI_SECTION);
    HCAL_HLX::LUMI_SECTION_HEADER* lumiheader = &(localSection->hdr);
    HCAL_HLX::LUMI_SUMMARY* lumisummary = &(localSection->lumiSummary);
    HCAL_HLX::LUMI_DETAIL* lumidetail = &(localSection->lumiDetail);
    hlxtree->SetBranchAddress("Header.",&lumiheader);
    hlxtree->SetBranchAddress("Summary.",&lumisummary);
    hlxtree->SetBranchAddress("Detail.",&lumidetail);

    size_t nentries=hlxtree->GetEntries();
    size_t ncmslumi=0;
    unsigned int totaldeadtime=0;
    //std::cout<<"processing total lumi lumisection "<<nentries<<std::endl;
    size_t lumisecid=0;
    unsigned int lumilumisecid=0;
    for(size_t i=0;i<nentries;++i){
      hlxtree->GetEntry(i);

      /*if(i==0){
	std::cout<<"Time stamp : "<<lumiheader->timestamp<<"\n";
	std::cout<<"Time stamp micro : "<<lumiheader->timestamp_micros<<"\n";
	std::cout<<"Run number : "<<lumiheader->runNumber<<"\n";
	std::cout<<"Section number : "<<lumiheader->sectionNumber<<"\n";
	std::cout<<"startOrbit : "<<lumiheader->startOrbit<<"\n";
	std::cout<<"numOrbit : "<<lumiheader->numOrbits<<"\n";
	std::cout<<"numBunches : "<<lumiheader->numBunches<<"\n";
	std::cout<<"numHLXs : "<<lumiheader->numHLXs<<"\n";
	std::cout<<"CMS Live : "<<lumiheader->bCMSLive<<"\n";
	std::cout<<"OC0 : "<<lumiheader->bOC0<<std::endl;
      }
      */
      // if not cms daq LS, skip
      if(!lumiheader->bCMSLive){
	std::cout<<"skipping non-CMS LS "<<lumiheader->sectionNumber<<std::endl;
	continue;
      }
      ++ncmslumi;
      ++lumisecid;
      if(allowForceFirstSince && i==0){ //if allowForceFirstSince and this is the head of the iov, then set the head to the begin of time
	runnumber=1;
      }else{
	runnumber=lumiheader->runNumber;
      }
      lumilumisecid=lumiheader->sectionNumber;
      edm::LuminosityBlockID lu(runnumber,lumisecid);
      cond::Time_t current=(cond::Time_t)(lu.value());
      lumi::LumiSectionData* l=new lumi::LumiSectionData;
      l->setLumiVersion(m_lumiversion);
      l->setStartOrbit((unsigned long long)lumiheader->startOrbit);
      l->setLumiAverage(lumisummary->InstantLumi);
      std::cout<<"lumisec "<<lumisecid<<" : lumilumisec "<<lumilumisecid<<" : inst lumi "<<lumisummary->InstantLumi<<std::endl;
      l->setLumiError(lumisummary->InstantLumiErr);      
      l->setLumiQuality(lumisummary->InstantLumiQlty);
            
      std::vector<lumi::BunchCrossingInfo> bxinfoET;
      bxinfoET.reserve(3564);
      for(size_t i=0;i<3564;++i){
	bxinfoET.push_back(lumi::BunchCrossingInfo(i+1,lumidetail->ETLumi[i],lumidetail->ETLumiErr[i],lumidetail->ETLumiQlty[i]));
      }
      l->setBunchCrossingData(bxinfoET,lumi::ET);
      
      std::vector<lumi::BunchCrossingInfo> bxinfoOCC1;
      std::vector<lumi::BunchCrossingInfo> bxinfoOCC2;
      bxinfoOCC1.reserve(3564);
      bxinfoOCC2.reserve(3564);
      for(size_t i=0;i<3564;++i){
	bxinfoOCC1.push_back(lumi::BunchCrossingInfo(i+1,lumidetail->OccLumi[0][i],lumidetail->OccLumiErr[0][i],lumidetail->OccLumiQlty[0][i]));
	bxinfoOCC2.push_back(lumi::BunchCrossingInfo(i+1,lumidetail->OccLumi[1][i],lumidetail->OccLumiErr[1][i],lumidetail->OccLumiQlty[1][i]));
      }
      l->setBunchCrossingData(bxinfoET,lumi::ET);
      l->setBunchCrossingData(bxinfoOCC1,lumi::OCCD1);
      l->setBunchCrossingData(bxinfoOCC2,lumi::OCCD2);
    }
    std::cout<<"total cms lumi "<<ncmslumi<<std::endl;
  }
  return m_filename+";"+m_lumiversion;
}

DEFINE_EDM_PLUGIN(lumi::LumiRetrieverFactory,lumi::MixedSource,"mixedsource");
 
