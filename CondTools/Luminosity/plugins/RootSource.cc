#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "CondCore/DBCommon/interface/Time.h"
#include "CondFormats/Luminosity/interface/LumiSectionData.h"
#include "CondTools/Luminosity/interface/LumiRetrieverFactory.h"
#include "RootSource.h"
#include <iostream>
#include "TFile.h"
#include "TTree.h"
lumi::RootSource::RootSource(const edm::ParameterSet& pset):LumiRetrieverBase(pset){
  m_filename=pset.getParameter<std::string>("lumiFileName");
  m_source=TFile::Open(m_filename.c_str(),"READ");
  std::cout<<"got file "<<m_source<<std::endl;
  m_source->ls();
  m_source->GetListOfKeys()->Print();

  TTree *runtree = (TTree*)m_source->Get("RunSummary");
  if(runtree){
    runtree->Print();
    std::cout<<"tot size bytes "<<runtree->GetTotBytes()<<std::endl;
    std::cout<<"n branches "<<runtree->GetNbranches()<<std::endl;
    std::cout<<"tot entries "<<runtree->GetEntries()<<std::endl;
  }

  TTree *rcmsconfig = (TTree*)m_source->Get("RCMSConfig");
  if(rcmsconfig){
    rcmsconfig->Print();
    std::cout<<"tot size bytes "<<rcmsconfig->GetTotBytes()<<std::endl;
    std::cout<<"n branches "<<rcmsconfig->GetNbranches()<<std::endl;
    std::cout<<"tot entries "<<rcmsconfig->GetEntries()<<std::endl;
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

  TTree *hlttrg = (TTree*)m_source->Get("HLTTrigger");
  if(hlttrg){
    hlttrg->Print();
    std::cout<<"tot size bytes "<<hlttrg->GetTotBytes()<<std::endl;
    std::cout<<"n branches "<<hlttrg->GetNbranches()<<std::endl;
    std::cout<<"tot entries "<<hlttrg->GetEntries()<<std::endl;
  }
  TTree *vdm = (TTree*)m_source->Get("VdMScan");
  if(vdm){
    vdm->Print();
    std::cout<<"tot size bytes "<<vdm->GetTotBytes()<<std::endl;
    std::cout<<"n branches "<<vdm->GetNbranches()<<std::endl;
    std::cout<<"tot entries "<<vdm->GetEntries()<<std::endl;
  }
}
void
lumi::RootSource::fill(std::vector< std::pair<lumi::LumiSectionData*,cond::Time_t> >& result){
}

DEFINE_EDM_PLUGIN(lumi::LumiRetrieverFactory,lumi::RootSource,"rootsource");
 
