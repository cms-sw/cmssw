#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "CondCore/DBCommon/interface/Time.h"
#include "CondFormats/Luminosity/interface/LumiSectionData.h"
#include "CondTools/Luminosity/interface/LumiRetrieverFactory.h"
#include "CondTools/Luminosity/interface/LumiDataStructures.h"
#include "RootSource.h"
#include <iostream>
#include "TFile.h"
#include "TTree.h"

lumi::RootSource::RootSource(const edm::ParameterSet& pset):LumiRetrieverBase(pset){
  m_filename=pset.getParameter<std::string>("lumiFileName");
  m_source=TFile::Open(m_filename.c_str(),"READ");
  m_source->GetListOfKeys()->Print();
  std::string::size_type idx,pos;
  idx=m_filename.rfind("_");
  pos=m_filename.rfind(".");
  if( idx == std::string::npos ){
  }
  m_lumiversion=m_filename.substr(idx+1,pos-idx-1);
}

void
lumi::RootSource::fill(std::vector< std::pair<lumi::LumiSectionData*,cond::Time_t> >& result){
  m_source->ls();

  TTree *hlxtree = (TTree*)m_source->Get("HLXData");
  TTree *l1tree = (TTree*)m_source->Get("L1Trigger");
  TTree *hlttree = (TTree*)m_source->Get("HLTrigger");
  unsigned int runnumber=0;
  if(hlxtree && l1tree && hlttree){
    hlxtree->Print();
    l1tree->Print();
    hlttree->Print();
    lumi::LUMI_SECTION_HEADER* lumiheader=0;
    lumi::LUMI_SUMMARY* lumisummary=0;
    lumi::LUMI_DETAIL* lumidetail=0;
    lumi::LEVEL1_TRIGGER* l1data=0;
    lumi::HLTRIGGER* hltdata=0;
    hlttree->SetBranchAddress("HLTrigger.",&hltdata);
    hlxtree->SetBranchAddress("Header.",&lumiheader);
    hlxtree->SetBranchAddress("Summary.",&lumisummary);
    hlxtree->SetBranchAddress("Detail.",&lumidetail);
    l1tree->SetBranchAddress("L1Trigger.",&l1data);
    
    size_t nentries=hlxtree->GetEntries();

    std::cout<<"total lumi lumisec "<<nentries<<std::endl;
    size_t lumisecid=0;
    for(size_t i=0;i<nentries;++i){
      hlxtree->GetEntry(i);
      l1tree->GetEntry(i);
      hlttree->GetEntry(i);
      
      runnumber=lumiheader->runNumber;
      lumisecid=lumiheader->sectionNumber;
      edm::LuminosityBlockID lu(runnumber,lumisecid);
      cond::Time_t current=(cond::Time_t)(lu.value());
      lumi::LumiSectionData* l=new lumi::LumiSectionData;
      l->setLumiVersion(m_lumiversion);
      l->setStartOrbit((unsigned long long)lumiheader->startOrbit);
      l->setLumiAverage(lumisummary->InstantLumi);
      l->setLumiError(lumisummary->InstantLumiErr);      
      l->setLumiQuality(lumisummary->InstantLumiQlty);
      l->setDeadFraction(lumisummary->DeadTimeNormalization);
      
      size_t hltsize=sizeof(hltdata->HLTPaths)/sizeof(lumi::HLT_PATH);
      std::cout<<"got "<<hltsize<<" hlt paths"<<std::endl;
      std::vector<lumi::HLTInfo> hltinfo;
      hltinfo.reserve(hltsize);
      //
      //fixme: missing hlt prescale from root, set to 1 for now
      //
      for( size_t ihlt=0; ihlt<hltsize; ++ihlt){
	lumi::HLTInfo hltperpath(std::string(hltdata->HLTPaths[ihlt].PathName),hltdata->HLTPaths[ihlt].L1Pass,hltdata->HLTPaths[ihlt].PAccept,1);
	std::cout<<"hltpath name "<<std::string(hltdata->HLTPaths[ihlt].PathName)<<std::endl;
	hltinfo.push_back(hltperpath);
      }
      
      //
      //fixme: missing l1 deadtimecount from root, set to 12387 for now
      //
      std::vector<lumi::TriggerInfo> triginfo;
      triginfo.reserve(192);
      size_t algotrgsize=sizeof(l1data->GTAlgo)/sizeof(lumi::LEVEL1_PATH);
      for( size_t itrg=0; itrg<algotrgsize; ++itrg ){
	lumi::TriggerInfo trgbit(l1data->GTAlgo[itrg].pathName,l1data->GTAlgo[itrg].counts,12387,l1data->GTAlgo[itrg].prescale);
	std::cout<<"l1path name "<<std::string(l1data->GTAlgo[itrg].pathName)<<std::endl;
	triginfo.push_back(trgbit);
      }
      std::cout<<"got "<<algotrgsize<<" algo trigger"<<std::endl;
      size_t techtrgsize=sizeof(l1data->GTTech)/sizeof(lumi::LEVEL1_PATH);
      for( size_t itrg=0; itrg<techtrgsize; ++itrg){
	lumi::TriggerInfo trgbit(l1data->GTTech[itrg].pathName,l1data->GTTech[itrg].counts,12387,l1data->GTTech[itrg].prescale);
	triginfo.push_back(trgbit);
      }
      std::cout<<"got "<<techtrgsize<<" tech trigger"<<std::endl;
      std::cout<<"bizzare cmsliveflag\t"<<(bool)lumiheader->bCMSLive<<std::endl;
      l->setHLTData(hltinfo);
      l->setTriggerData(triginfo);

      
      result.push_back(std::make_pair<lumi::LumiSectionData*,cond::Time_t>(l,current));
    }
  }
}

DEFINE_EDM_PLUGIN(lumi::LumiRetrieverFactory,lumi::RootSource,"rootsource");
 
