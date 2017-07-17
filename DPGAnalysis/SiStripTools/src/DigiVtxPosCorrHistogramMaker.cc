#include "DPGAnalysis/SiStripTools/interface/DigiVtxPosCorrHistogramMaker.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TH2F.h"
#include "TProfile.h"

#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "DPGAnalysis/SiStripTools/interface/SiStripTKNumbers.h"


DigiVtxPosCorrHistogramMaker::DigiVtxPosCorrHistogramMaker(edm::ConsumesCollector&& iC):
  m_mcvtxcollectionToken(iC.consumes<edm::HepMCProduct>(edm::InputTag("generatorSmeared"))), m_hitname(), m_nbins(500), m_scalefact(), m_binmax(), m_labels(),
  m_nmultvsvtxpos(), m_nmultvsvtxposprof(), m_subdirs() { }

DigiVtxPosCorrHistogramMaker::DigiVtxPosCorrHistogramMaker(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& iC):
  m_mcvtxcollectionToken(iC.consumes<edm::HepMCProduct>(iConfig.getParameter<edm::InputTag>("mcVtxCollection"))),
  m_hitname(iConfig.getUntrackedParameter<std::string>("hitName","digi")),
  m_nbins(iConfig.getUntrackedParameter<int>("numberOfBins",500)),
  m_scalefact(iConfig.getUntrackedParameter<int>("scaleFactor",5)),
  m_labels(), m_nmultvsvtxpos(), m_nmultvsvtxposprof(), m_subdirs()
{

  std::vector<edm::ParameterSet>
    wantedsubds(iConfig.getUntrackedParameter<std::vector<edm::ParameterSet> >("wantedSubDets",std::vector<edm::ParameterSet>()));

  for(std::vector<edm::ParameterSet>::iterator ps=wantedsubds.begin();ps!=wantedsubds.end();++ps) {
    m_labels[ps->getParameter<unsigned int>("detSelection")] = ps->getParameter<std::string>("detLabel");
    m_binmax[ps->getParameter<unsigned int>("detSelection")] = ps->getParameter<int>("binMax");
  }


}


DigiVtxPosCorrHistogramMaker::~DigiVtxPosCorrHistogramMaker() {

  for(std::map<unsigned int,std::string>::const_iterator lab=m_labels.begin();lab!=m_labels.end();lab++) {

    const unsigned int i = lab->first; const std::string slab = lab->second;

    delete m_subdirs[i];
  }

}



void DigiVtxPosCorrHistogramMaker::book(const std::string dirname, const std::map<unsigned int, std::string>& labels) {

  m_labels = labels;
  book(dirname);

}

void DigiVtxPosCorrHistogramMaker::book(const std::string dirname) {

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
      sprintf(name,"n%sdigivsvtxpos",slab.c_str());
      sprintf(title,"%s %s multiplicity vx MC vertex z position",slab.c_str(),m_hitname.c_str());
      m_nmultvsvtxpos[i] = m_subdirs[i]->make<TH2F>(name,title,200,-20.,20.,m_nbins,0.,m_binmax[i]/(m_scalefact*m_nbins)*m_nbins);
      m_nmultvsvtxpos[i]->GetXaxis()->SetTitle("MC vertex z position (cm)"); m_nmultvsvtxpos[i]->GetYaxis()->SetTitle("Number of Hits");
      sprintf(name,"n%sdigivsvtxposprof",slab.c_str());
      m_nmultvsvtxposprof[i] = m_subdirs[i]->make<TProfile>(name,title,200,-20.,20.);
      m_nmultvsvtxposprof[i]->GetXaxis()->SetTitle("MC vertex z position (cm)");    m_nmultvsvtxposprof[i]->GetYaxis()->SetTitle("Number of Hits");

    }

  }


}

void DigiVtxPosCorrHistogramMaker::beginRun(const unsigned int nrun) {


}

void DigiVtxPosCorrHistogramMaker::fill(const edm::Event& iEvent, const std::map<unsigned int,int>& ndigi) {

  // main interaction part

  edm::Handle< edm::HepMCProduct > EvtHandle ;
  iEvent.getByToken(m_mcvtxcollectionToken, EvtHandle ) ;

  if(EvtHandle.isValid()) {

    const HepMC::GenEvent* Evt = EvtHandle->GetEvent();

    // get the first vertex

    if(Evt->vertices_begin() != Evt->vertices_end()) {

      double vtxz = (*Evt->vertices_begin())->point3d().z()/10.;

      for(std::map<unsigned int,int>::const_iterator digi=ndigi.begin();digi!=ndigi.end();digi++) {
	if(m_labels.find(digi->first) != m_labels.end()) {
	  const unsigned int i=digi->first;
	  m_nmultvsvtxpos[i]->Fill(vtxz,digi->second);
	  m_nmultvsvtxposprof[i]->Fill(vtxz,digi->second);
	}
      }
    }
  }
}


