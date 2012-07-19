#include "DPGAnalysis/SiStripTools/interface/DigiLumiCorrHistogramMaker.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/Luminosity/interface/LumiDetails.h"
#include "TH2F.h"
#include "TProfile.h"

#include "DPGAnalysis/SiStripTools/interface/SiStripTKNumbers.h"
#include "DPGAnalysis/SiStripTools/interface/RunHistogramManager.h"


DigiLumiCorrHistogramMaker::DigiLumiCorrHistogramMaker():
  m_lumiProducer("lumiProducer"),   m_fhm(),  m_runHisto(false),
  m_hitname(), m_nbins(500), m_scalefact(), m_maxlumi(10.), m_binmax(), m_labels(), m_nmultvslumi(), m_nmultvslumiprof(), m_subdirs() { }

DigiLumiCorrHistogramMaker::DigiLumiCorrHistogramMaker(const edm::ParameterSet& iConfig):
  m_lumiProducer(iConfig.getParameter<edm::InputTag>("lumiProducer")),
  m_fhm(),
  m_runHisto(iConfig.getUntrackedParameter<bool>("runHisto",false)),
  m_hitname(iConfig.getUntrackedParameter<std::string>("hitName","digi")),
  m_nbins(iConfig.getUntrackedParameter<int>("numberOfBins",500)),
  m_scalefact(iConfig.getUntrackedParameter<int>("scaleFactor",5)),
  m_maxlumi(iConfig.getUntrackedParameter<double>("maxLumi",10.)),
  m_labels(), m_nmultvslumi(), m_nmultvslumiprof(), m_subdirs()
{ 

  std::vector<edm::ParameterSet> 
    wantedsubds(iConfig.getUntrackedParameter<std::vector<edm::ParameterSet> >("wantedSubDets",std::vector<edm::ParameterSet>()));
  
  for(std::vector<edm::ParameterSet>::iterator ps=wantedsubds.begin();ps!=wantedsubds.end();++ps) {
    m_labels[ps->getParameter<unsigned int>("detSelection")] = ps->getParameter<std::string>("detLabel");
    m_binmax[ps->getParameter<unsigned int>("detSelection")] = ps->getParameter<int>("binMax");
  }
  

}


DigiLumiCorrHistogramMaker::~DigiLumiCorrHistogramMaker() {

  for(std::map<unsigned int,std::string>::const_iterator lab=m_labels.begin();lab!=m_labels.end();lab++) {
    
    const unsigned int i = lab->first; const std::string slab = lab->second;
    
    delete m_subdirs[i];
    delete m_fhm[i];
  }
  
}



void DigiLumiCorrHistogramMaker::book(const std::string dirname, const std::map<unsigned int, std::string>& labels) {

  m_labels = labels;
  book(dirname);

}

void DigiLumiCorrHistogramMaker::book(const std::string dirname) {

  edm::Service<TFileService> tfserv;
  TFileDirectory subev = tfserv->mkdir(dirname);

  SiStripTKNumbers trnumb;
  
  edm::LogInfo("NumberOfBins") << "Number of Bins: " << m_nbins;
  edm::LogInfo("ScaleFactors") << "y-axis range scale factor: " << m_scalefact;
  edm::LogInfo("MaxLumi") << "max lumi value: " << m_maxlumi;
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
    m_fhm[i] = new RunHistogramManager(true);

    if(m_subdirs[i]) {
      sprintf(name,"n%sdigivslumi",slab.c_str());
      sprintf(title,"%s %s multiplicity vs BX lumi",slab.c_str(),m_hitname.c_str());
      m_nmultvslumi[i] = m_subdirs[i]->make<TH2F>(name,title,250,0.,m_maxlumi,m_nbins,0.,(1+m_binmax[i]/(m_scalefact*m_nbins))*m_nbins);
      m_nmultvslumi[i]->GetXaxis()->SetTitle("BX lumi [10^{30}cm^{-2}s^{-1}]");    m_nmultvslumi[i]->GetYaxis()->SetTitle("Number of Hits");
      sprintf(name,"n%sdigivslumiprof",slab.c_str());
      m_nmultvslumiprof[i] = m_subdirs[i]->make<TProfile>(name,title,250,0.,m_maxlumi);
      m_nmultvslumiprof[i]->GetXaxis()->SetTitle("BX lumi [10^{30}cm^{-2}s^{-1}]");    m_nmultvslumiprof[i]->GetYaxis()->SetTitle("Number of Hits");
      
      if(m_runHisto) {
	edm::LogInfo("RunHistos") << "Pseudo-booking run histos " << slab.c_str();
	sprintf(name,"n%sdigivslumivsbxprofrun",slab.c_str());
	sprintf(title,"%s %s multiplicity vs BX lumi vs BX",slab.c_str(),m_hitname.c_str());
	m_nmultvslumivsbxprofrun[i] = m_fhm[i]->makeTProfile2D(name,title,3564,-0.5,3563.5,250,0.,m_maxlumi);
      }
    }

  }


}

void DigiLumiCorrHistogramMaker::beginRun(const edm::Run& iRun) {

  edm::Service<TFileService> tfserv;


  for(std::map<unsigned int,std::string>::const_iterator lab=m_labels.begin();lab!=m_labels.end();++lab) {
    const int i = lab->first; const std::string slab = lab->second;
    m_fhm[i]->beginRun(iRun,*m_subdirs[i]);
    if(m_runHisto) {
      (*m_nmultvslumivsbxprofrun[i])->GetXaxis()->SetTitle("BX");    
      (*m_nmultvslumivsbxprofrun[i])->GetYaxis()->SetTitle("BX lumi [10^{30}cm^{-2}s^{-1}]");
    }
  }


}

void DigiLumiCorrHistogramMaker::fill(const edm::Event& iEvent, const std::map<unsigned int,int>& ndigi) {
  
  edm::Handle<LumiDetails> ld;
  iEvent.getLuminosityBlock().getByLabel(m_lumiProducer,ld);

  if(ld.isValid()) {
    if(ld->isValid()) {
      float bxlumi = ld->lumiValue(LumiDetails::kOCC1,iEvent.bunchCrossing())*6.37;

      for(std::map<unsigned int,int>::const_iterator digi=ndigi.begin();digi!=ndigi.end();digi++) {
	if(m_labels.find(digi->first) != m_labels.end()) {
	  const unsigned int i=digi->first;
	  m_nmultvslumi[i]->Fill(bxlumi,digi->second);
	  m_nmultvslumiprof[i]->Fill(bxlumi,digi->second);

	  if(m_nmultvslumivsbxprofrun[i] && *m_nmultvslumivsbxprofrun[i]) (*m_nmultvslumivsbxprofrun[i])->Fill(iEvent.bunchCrossing(),bxlumi,digi->second);

	}
      }
    }
  }
}
    
