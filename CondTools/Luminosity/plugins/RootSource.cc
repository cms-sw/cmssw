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
  /*TTree *runtree = (TTree*)m_source->Get("RunSummary");
    if(runtree){
    runtree->Print();
    //std::cout<<"tot size bytes "<<runtree->GetTotBytes()<<std::endl;
    lumi::RUN_SUMMARY* runsummarydata=0;
    runtree->SetBranchAddress("RunSummary.",&runsummarydata);
    size_t nentries=runtree->GetEntries();
    for(size_t i=0;i<nentries;++i){
    runtree->GetEntry(i);
      runnumber=runsummarydata->runNumber;
      }
  }
  */

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
    lumi::HLTrigger* hltdata=0;
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
      l->setDeadFraction(lumisummary->DeadtimeNormalization);
      std::cout<<"bizzar cmsliveflag\t"<<(bool)lumiheader->bCMSLive<<std::endl;
      result.push_back(std::make_pair<lumi::LumiSectionData*,cond::Time_t>(l,current));
    }
  }
}

DEFINE_EDM_PLUGIN(lumi::LumiRetrieverFactory,lumi::RootSource,"rootsource");
 
