#include "DPGAnalysis/SiStripTools/interface/DigiVertexCorrHistogramMaker.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TH2F.h"

#include "DPGAnalysis/SiStripTools/interface/SiStripTKNumbers.h"


DigiVertexCorrHistogramMaker::DigiVertexCorrHistogramMaker():
  m_hitname(), m_nbins(500), m_scalefact(), m_binmax(), m_labels(), m_nmultvsnvtx() { }

DigiVertexCorrHistogramMaker::DigiVertexCorrHistogramMaker(const edm::ParameterSet& iConfig):
  m_hitname(iConfig.getUntrackedParameter<std::string>("hitName","digi")),
  m_nbins(iConfig.getUntrackedParameter<int>("numberOfBins",500)),
  m_scalefact(iConfig.getUntrackedParameter<int>("scaleFactor",5)),
  m_labels(), m_nmultvsnvtx(), m_subdirs()
{ 

  std::vector<edm::ParameterSet> 
    wantedsubds(iConfig.getUntrackedParameter<std::vector<edm::ParameterSet> >("wantedSubDets",std::vector<edm::ParameterSet>()));
  
  for(std::vector<edm::ParameterSet>::iterator ps=wantedsubds.begin();ps!=wantedsubds.end();++ps) {
    m_labels[ps->getParameter<unsigned int>("detSelection")] = ps->getParameter<std::string>("detLabel");
    m_binmax[ps->getParameter<unsigned int>("detSelection")] = ps->getParameter<int>("binMax");
  }
  

}


DigiVertexCorrHistogramMaker::~DigiVertexCorrHistogramMaker() {

  for(std::map<unsigned int,std::string>::const_iterator lab=m_labels.begin();lab!=m_labels.end();lab++) {
    
    const unsigned int i = lab->first; const std::string slab = lab->second;
    
    delete m_subdirs[i];
  }
  
}



void DigiVertexCorrHistogramMaker::book(const std::string dirname, const std::map<unsigned int, std::string>& labels) {

  m_labels = labels;
  book(dirname);

}

void DigiVertexCorrHistogramMaker::book(const std::string dirname) {

  edm::Service<TFileService> tfserv;
  TFileDirectory subev = tfserv->mkdir(dirname);

  SiStripTKNumbers trnumb;
  
  edm::LogInfo("NumberOfBins") << "Number of Bins: " << m_nbins;
  edm::LogInfo("ScaleFactors") << "y-axis range scale factor: " << m_scalefact;
  edm::LogInfo("BinMaxValue") << "Setting bin max values";

  for(std::map<unsigned int,std::string>::const_iterator lab=m_labels.begin();lab!=m_labels.end();lab++) {
    
    const unsigned int i = lab->first; const std::string slab = lab->second;
    
    if(m_binmax.find(i)==m_binmax.end()) {
      edm::LogVerbatim("NotConfiguredBinMax") << "Bin max for " << lab->second 
					      << " not configured: " << trnumb.nstrips(i) << " used";
      m_binmax[i] = trnumb.nstrips(i);
    }
 
    edm::LogVerbatim("BinMaxValue") << "Bin max for " << lab->second << " is " << m_binmax[i];

  }

  for(std::map<unsigned int,std::string>::const_iterator lab=m_labels.begin();lab!=m_labels.end();++lab) {

    const int i = lab->first; const std::string slab = lab->second;

    char name[200];
    char title[500];

    m_subdirs[i] = new TFileDirectory(subev.mkdir(slab.c_str()));

    if(m_subdirs[i]) {
      sprintf(name,"n%sdigivsnvtx",slab.c_str());
      sprintf(title,"%s %s multiplicity vs Nvtx",slab.c_str(),m_hitname.c_str());
      m_nmultvsnvtx[i] = m_subdirs[i]->make<TH2F>(name,title,60,-0.5,59.5,m_nbins,0.,m_binmax[i]/(m_scalefact*m_nbins)*m_nbins);
      m_nmultvsnvtx[i]->GetXaxis()->SetTitle("Number of Vertices");    m_nmultvsnvtx[i]->GetYaxis()->SetTitle("Number of Hits");
      
    }

  }


}

void DigiVertexCorrHistogramMaker::beginRun(const unsigned int nrun) {


}

void DigiVertexCorrHistogramMaker::fill(const unsigned int nvtx, const std::map<unsigned int,int>& ndigi) {
  
  for(std::map<unsigned int,int>::const_iterator digi=ndigi.begin();digi!=ndigi.end();digi++) {
    
    if(m_labels.find(digi->first) != m_labels.end()) {
 
      const unsigned int i=digi->first;
      m_nmultvsnvtx[i]->Fill(nvtx,digi->second);

    }
    
  }
}

