#include "DPGAnalysis/SiStripTools/interface/DigiVertexCorrHistogramMaker.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TH2F.h"
#include "TProfile.h"

#include "DPGAnalysis/SiStripTools/interface/SiStripTKNumbers.h"
#include "DPGAnalysis/SiStripTools/interface/RunHistogramManager.h"


DigiVertexCorrHistogramMaker::DigiVertexCorrHistogramMaker():
  m_fhm(),
  m_runHisto(false),
  m_hitname(), m_nbins(500), m_scalefact(), m_maxnvtx(60), m_binmax(), m_labels(), m_nmultvsnvtx(), m_nmultvsnvtxprof(), m_nmultvsnvtxvsbxprofrun(), m_subdirs() { }

DigiVertexCorrHistogramMaker::DigiVertexCorrHistogramMaker(const edm::ParameterSet& iConfig):
  m_fhm(),
  m_runHisto(iConfig.getUntrackedParameter<bool>("runHisto",false)),
  m_hitname(iConfig.getUntrackedParameter<std::string>("hitName","digi")),
  m_nbins(iConfig.getUntrackedParameter<int>("numberOfBins",500)),
  m_scalefact(iConfig.getUntrackedParameter<int>("scaleFactor",5)),
  m_maxnvtx(iConfig.getUntrackedParameter<int>("maxNvtx",60)),
  m_labels(), m_nmultvsnvtx(), m_nmultvsnvtxprof(), m_subdirs()
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
    delete m_fhm[i];
  }

}



void DigiVertexCorrHistogramMaker::book(const std::string dirname, const std::map<unsigned int, std::string>& labels, edm::ConsumesCollector&& iC) {

  m_labels = labels;
  book(dirname, iC);

}

void DigiVertexCorrHistogramMaker::book(const std::string dirname, edm::ConsumesCollector& iC) {

  edm::Service<TFileService> tfserv;
  TFileDirectory subev = tfserv->mkdir(dirname);

  SiStripTKNumbers trnumb;

  edm::LogInfo("NumberOfBins") << "Number of Bins: " << m_nbins;
  edm::LogInfo("ScaleFactors") << "y-axis range scale factor: " << m_scalefact;
  edm::LogInfo("MaxNvtx") << "maximum number of vertices: " << m_maxnvtx;
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
    m_fhm[i] = new RunHistogramManager(iC, true);

    if(m_subdirs[i]) {
      sprintf(name,"n%sdigivsnvtx",slab.c_str());
      sprintf(title,"%s %s multiplicity vs Nvtx",slab.c_str(),m_hitname.c_str());
      m_nmultvsnvtx[i] = m_subdirs[i]->make<TH2F>(name,title,m_maxnvtx,-0.5,m_maxnvtx-0.5,m_nbins,0.,(1+m_binmax[i]/(m_scalefact*m_nbins))*m_nbins);
      m_nmultvsnvtx[i]->GetXaxis()->SetTitle("Number of Vertices");    m_nmultvsnvtx[i]->GetYaxis()->SetTitle("Number of Hits");
      sprintf(name,"n%sdigivsnvtxprof",slab.c_str());
      m_nmultvsnvtxprof[i] = m_subdirs[i]->make<TProfile>(name,title,m_maxnvtx,-0.5,m_maxnvtx-0.5);
      m_nmultvsnvtxprof[i]->GetXaxis()->SetTitle("Number of Vertices");    m_nmultvsnvtxprof[i]->GetYaxis()->SetTitle("Number of Hits");

      if(m_runHisto) {
	edm::LogInfo("RunHistos") << "Pseudo-booking run histos " << slab.c_str();
	sprintf(name,"n%sdigivsnvtxvsbxprofrun",slab.c_str());
	sprintf(title,"%s %s multiplicity vs Nvtx vs BX",slab.c_str(),m_hitname.c_str());
	m_nmultvsnvtxvsbxprofrun[i] = m_fhm[i]->makeTProfile2D(name,title,3564,-0.5,3563.5,m_maxnvtx,-0.5,m_maxnvtx-0.5);
      }

    }

  }


}

void DigiVertexCorrHistogramMaker::beginRun(const edm::Run& iRun) {

  edm::Service<TFileService> tfserv;


  for(std::map<unsigned int,std::string>::const_iterator lab=m_labels.begin();lab!=m_labels.end();++lab) {
    const int i = lab->first; const std::string slab = lab->second;
    m_fhm[i]->beginRun(iRun,*m_subdirs[i]);
    if(m_runHisto) {
      (*m_nmultvsnvtxvsbxprofrun[i])->GetXaxis()->SetTitle("BX");    (*m_nmultvsnvtxvsbxprofrun[i])->GetYaxis()->SetTitle("Nvertices");
    }
  }

}

void DigiVertexCorrHistogramMaker::fill(const edm::Event& iEvent, const unsigned int nvtx, const std::map<unsigned int,int>& ndigi) {


  edm::Service<TFileService> tfserv;


  for(std::map<unsigned int,int>::const_iterator digi=ndigi.begin();digi!=ndigi.end();digi++) {

    if(m_labels.find(digi->first) != m_labels.end()) {

      const unsigned int i=digi->first;
      m_nmultvsnvtx[i]->Fill(nvtx,digi->second);
      m_nmultvsnvtxprof[i]->Fill(nvtx,digi->second);

      if(m_nmultvsnvtxvsbxprofrun[i] && *m_nmultvsnvtxvsbxprofrun[i]) (*m_nmultvsnvtxvsbxprofrun[i])->Fill(iEvent.bunchCrossing(),nvtx,digi->second);

    }

  }
}

