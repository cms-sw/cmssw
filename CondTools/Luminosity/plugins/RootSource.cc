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
    std::cout<<"tot entries "<<runtree->GetEntries()<<std::endl;
  }

  TTree *hlxdata = (TTree*)m_source->Get("HLXData");
  if(hlxdata){
    hlxdata->Print();
    std::cout<<"tot size bytes "<<hlxdata->GetTotBytes()<<std::endl;
    std::cout<<"n branches "<<hlxdata->GetNbranches()<<std::endl;
    std::cout<<"tot entries "<<hlxdata->GetEntries()<<std::endl;
  }

  TTree *l1trg = (TTree*)m_source->Get("L1Trigger");
  if(l1trg){
    l1trg->Print();
    std::cout<<"tot size bytes "<<l1trg->GetTotBytes()<<std::endl;
    std::cout<<"n branches "<<l1trg->GetNbranches()<<std::endl;
    std::cout<<"tot entries "<<l1trg->GetEntries()<<std::endl;
  }

  TTree *hlttrg = (TTree*)m_source->Get("HLTrigger");
  if(hlttrg){
    hlttrg->Print();
    std::cout<<"tot size bytes "<<hlttrg->GetTotBytes()<<std::endl;
    std::cout<<"n branches "<<hlttrg->GetNbranches()<<std::endl;
    std::cout<<"tot entries "<<hlttrg->GetEntries()<<std::endl;
    
  }
}
void
lumi::RootSource::fill(std::vector< std::pair<lumi::LumiSectionData*,cond::Time_t> >& result){
}

DEFINE_EDM_PLUGIN(lumi::LumiRetrieverFactory,lumi::RootSource,"rootsource");
 
