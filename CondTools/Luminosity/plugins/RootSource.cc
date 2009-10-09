#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "CondCore/DBCommon/interface/Time.h"
#include "CondFormats/Luminosity/interface/LumiSectionData.h"
#include "CondTools/Luminosity/interface/LumiRetrieverFactory.h"
#include "LumiDataStructures.h"
#include "RootSource.h"
#include <memory>
#include <iostream>
#include "TFile.h"
#include "TTree.h"

lumi::RootSource::RootSource(const edm::ParameterSet& pset):LumiRetrieverBase(pset){
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
lumi::RootSource::fill(std::vector< std::pair<lumi::LumiSectionData*,cond::Time_t> >& result , bool allowForceFirstSince ){
  m_source->ls();
  TTree *hlxtree = (TTree*)m_source->Get("HLXData");
  TTree *l1tree = (TTree*)m_source->Get("L1Trigger");
  TTree *hlttree = (TTree*)m_source->Get("HLTrigger");
  unsigned int runnumber=0;
  if(hlxtree && l1tree && hlttree){
    hlxtree->Print();
    l1tree->Print();
    hlttree->Print();
    std::auto_ptr<HCAL_HLX::LUMI_SECTION> localSection(new HCAL_HLX::LUMI_SECTION);
    HCAL_HLX::LUMI_SECTION_HEADER* lumiheader = &(localSection->hdr);
    HCAL_HLX::LUMI_SUMMARY* lumisummary = &(localSection->lumiSummary);
    HCAL_HLX::LUMI_DETAIL* lumidetail = &(localSection->lumiDetail);
    HCAL_HLX::LEVEL1_TRIGGER* l1data=0;
    HCAL_HLX::HLTRIGGER* hltdata=0;
    hlttree->SetBranchAddress("HLTrigger.",&hltdata);
    hlxtree->SetBranchAddress("Header.",&lumiheader);
    hlxtree->SetBranchAddress("Summary.",&lumisummary);
    hlxtree->SetBranchAddress("Detail.",&lumidetail);
    l1tree->SetBranchAddress("L1Trigger.",&l1data);
    
    size_t nentries=hlxtree->GetEntries();
    std::cout<<"processing total lumi lumisection "<<nentries<<std::endl;
    size_t lumisecid=0;
    for(size_t i=0;i<nentries;++i){
      hlxtree->GetEntry(i);
      l1tree->GetEntry(i);
      hlttree->GetEntry(i);
      if(i==0){
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
      if(allowForceFirstSince && i==0){ //if allowForceFirstSince and this is the head of the iov, then set the head to the begin of time
	runnumber=1;
	lumisecid=1;
      }else{
	runnumber=lumiheader->runNumber;
	lumisecid=lumiheader->sectionNumber;
      }
      edm::LuminosityBlockID lu(runnumber,lumisecid);
      cond::Time_t current=(cond::Time_t)(lu.value());
      lumi::LumiSectionData* l=new lumi::LumiSectionData;
      l->setLumiVersion(m_lumiversion);
      l->setStartOrbit((unsigned long long)lumiheader->startOrbit);
      l->setLumiAverage(lumisummary->InstantLumi);
      l->setLumiError(lumisummary->InstantLumiErr);      
      l->setLumiQuality(lumisummary->InstantLumiQlty);
      l->setDeadFraction(lumisummary->DeadTimeNormalization);
      
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
      
      size_t hltsize=sizeof(hltdata->HLTPaths)/sizeof(HCAL_HLX::HLT_PATH);
      std::vector<lumi::HLTInfo> hltinfo;
      hltinfo.reserve(hltsize);

      for( size_t ihlt=0; ihlt<hltsize; ++ihlt){
	lumi::HLTInfo hltperpath(std::string(hltdata->HLTPaths[ihlt].PathName),hltdata->HLTPaths[ihlt].L1Pass,hltdata->HLTPaths[ihlt].PAccept,hltdata->HLTPaths[ihlt].Prescale);
	hltinfo.push_back(hltperpath);
      }
      
      std::vector<lumi::TriggerInfo> triginfo;
      triginfo.reserve(192);
      size_t algotrgsize=sizeof(l1data->GTAlgo)/sizeof(HCAL_HLX::LEVEL1_PATH);
      for( size_t itrg=0; itrg<algotrgsize; ++itrg ){
	lumi::TriggerInfo trgbit(l1data->GTAlgo[itrg].pathName,l1data->GTAlgo[itrg].counts,l1data->GTAlgo[itrg].deadtimecount,l1data->GTAlgo[itrg].prescale);
	triginfo.push_back(trgbit);
      }
      size_t techtrgsize=sizeof(l1data->GTTech)/sizeof(HCAL_HLX::LEVEL1_PATH);
      for( size_t itrg=0; itrg<techtrgsize; ++itrg){
	lumi::TriggerInfo trgbit(l1data->GTTech[itrg].pathName,l1data->GTTech[itrg].counts,l1data->GTTech[itrg].deadtimecount,l1data->GTTech[itrg].prescale);
	triginfo.push_back(trgbit);
      }
      //std::cout<<"bizzare cmsliveflag\t"<<(bool)lumiheader->bCMSLive<<std::endl;
      l->setHLTData(hltinfo);
      l->setTriggerData(triginfo);

      result.push_back(std::make_pair<lumi::LumiSectionData*,cond::Time_t>(l,current));
    }
  }
  return m_filename+";"+m_lumiversion;
}

DEFINE_EDM_PLUGIN(lumi::LumiRetrieverFactory,lumi::RootSource,"rootsource");
 
