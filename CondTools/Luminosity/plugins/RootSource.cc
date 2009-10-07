#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "CondCore/DBCommon/interface/Time.h"
#include "CondFormats/Luminosity/interface/LumiSectionData.h"
#include "CondTools/Luminosity/interface/LumiRetrieverFactory.h"
#include "CondTools/Luminosity/interface/LumiStructures.hh"
#include "RootSource.h"
#include <iostream>
#include "TFile.h"
#include "TTree.h"

lumi::RootSource::RootSource(const edm::ParameterSet& pset):LumiRetrieverBase(pset){
  m_filename=pset.getParameter<std::string>("lumiFileName");
  m_source=TFile::Open(m_filename.c_str(),"READ");
  m_source->ls();
  m_source->GetListOfKeys()->Print();

  TTree *runtree = (TTree*)m_source->Get("RunSummary");
  if(runtree){
    runtree->Print();
    std::cout<<"tot size bytes "<<runtree->GetTotBytes()<<std::endl;
    std::cout<<"n branches "<<runtree->GetNbranches()<<std::endl;
    HCAL_HLX::RUN_SUMMARY* runsummarydata=0;
    runtree->SetBranchAddress("RunSummary.",&runsummarydata);
    size_t nentries=runtree->GetEntries();
    for(size_t i=0;i<nentries;++i){
      runtree->GetEntry(i);
      std::cout<<runsummarydata->runNumber<<std::endl;
    }
  }

  TTree *hlxtree = (TTree*)m_source->Get("HLXData");
  if(hlxtree){
    hlxtree->Print();
    std::cout<<"tot size bytes "<<hlxtree->GetTotBytes()<<std::endl;
    std::cout<<"n branches "<<hlxtree->GetNbranches()<<std::endl;
    //HCAL_HLX::LUMI_SECTION* hlxdata=0;
    HCAL_HLX::LUMI_SECTION_HEADER* lumiheader=0;
    HCAL_HLX::LUMI_SUMMARY* lumisummary=0;
    HCAL_HLX::LUMI_DETAIL* lumidetail=0;
    hlxtree->SetBranchAddress("Header.",&lumiheader);
    hlxtree->SetBranchAddress("Summary.",&lumisummary);
    hlxtree->SetBranchAddress("Detail.",&lumidetail);
    size_t nentries=hlxtree->GetEntries();
    std::cout<<"hlxdata entries "<<nentries<<std::endl;
    for(size_t i=0;i<nentries;++i){
      hlxtree->GetEntry(i);
      std::cout<<lumiheader->runNumber<<std::endl;
      std::cout<<lumiheader->sectionNumber<<std::endl;
      std::cout<<lumiheader->startOrbit<<std::endl;
      std::cout<<lumisummary->InstantLumi<<std::endl;
    }
  }

  TTree *l1tree = (TTree*)m_source->Get("L1Trigger");
  if(l1tree){
    l1tree->Print();
    std::cout<<"tot size bytes "<<l1tree->GetTotBytes()<<std::endl;
    std::cout<<"n branches "<<l1tree->GetNbranches()<<std::endl;
    HCAL_HLX::LEVEL1_TRIGGER* l1data=0;
    l1tree->SetBranchAddress("L1Trigger.",&l1data);
    size_t nentries=l1tree->GetEntries();
    std::cout<<"l1tree entries "<<nentries<<std::endl;
  }

  TTree *hlttree = (TTree*)m_source->Get("HLTrigger");
  if(hlttree){
    hlttree->Print();
    std::cout<<"tot size bytes "<<hlttree->GetTotBytes()<<std::endl;
    std::cout<<"n branches "<<hlttree->GetNbranches()<<std::endl;
    HCAL_HLX::HLTrigger* hltdata=0;
    hlttree->SetBranchAddress("HLTrigger.",&hltdata);
    size_t nentries=hlttree->GetEntries();
    std::cout<<"hlttree entries "<<nentries<<std::endl;
  }
}
void
lumi::RootSource::fill(std::vector< std::pair<lumi::LumiSectionData*,cond::Time_t> >& result){
}

DEFINE_EDM_PLUGIN(lumi::LumiRetrieverFactory,lumi::RootSource,"rootsource");
 
