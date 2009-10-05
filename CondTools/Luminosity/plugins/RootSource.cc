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
  std::cout<<"file size "<<m_source->GetBytesRead()<<std::endl;
  m_source->ls();
  TTree *t = (TTree*)m_source->Get("RunSummary");
  t->Print();

}
void
lumi::RootSource::fill(std::vector< std::pair<lumi::LumiSectionData*,cond::Time_t> >& result){
}

DEFINE_EDM_PLUGIN(lumi::LumiRetrieverFactory,lumi::RootSource,"rootsource");
 
