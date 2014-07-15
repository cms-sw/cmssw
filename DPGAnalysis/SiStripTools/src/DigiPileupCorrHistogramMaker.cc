#include "DPGAnalysis/SiStripTools/interface/DigiPileupCorrHistogramMaker.h"
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


DigiPileupCorrHistogramMaker::DigiPileupCorrHistogramMaker(edm::ConsumesCollector&& iC):
  m_pileupcollectionToken(iC.consumes<std::vector<PileupSummaryInfo> >(edm::InputTag("addPileupInfo"))), m_useVisibleVertices(false), m_hitname(), m_nbins(500), m_scalefact(), m_binmax(), m_labels(),
  m_nmultvsmclumi(), m_nmultvsmclumiprof(), m_nmultvsmcnvtx(), m_nmultvsmcnvtxprof(), m_subdirs() { }

DigiPileupCorrHistogramMaker::DigiPileupCorrHistogramMaker(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& iC):
  m_pileupcollectionToken(iC.consumes<std::vector<PileupSummaryInfo> >(iConfig.getParameter<edm::InputTag>("pileupSummaryCollection"))),
  m_useVisibleVertices(iConfig.getParameter<bool>("useVisibleVertices")),
  m_hitname(iConfig.getUntrackedParameter<std::string>("hitName","digi")),
  m_nbins(iConfig.getUntrackedParameter<int>("numberOfBins",500)),
  m_scalefact(iConfig.getUntrackedParameter<int>("scaleFactor",5)),
  m_labels(), m_nmultvsmclumi(), m_nmultvsmclumiprof(), m_nmultvsmcnvtx(), m_nmultvsmcnvtxprof(), m_subdirs()
{

  std::vector<edm::ParameterSet>
    wantedsubds(iConfig.getUntrackedParameter<std::vector<edm::ParameterSet> >("wantedSubDets",std::vector<edm::ParameterSet>()));

  for(std::vector<edm::ParameterSet>::iterator ps=wantedsubds.begin();ps!=wantedsubds.end();++ps) {
    m_labels[ps->getParameter<unsigned int>("detSelection")] = ps->getParameter<std::string>("detLabel");
    m_binmax[ps->getParameter<unsigned int>("detSelection")] = ps->getParameter<int>("binMax");
  }


}


DigiPileupCorrHistogramMaker::~DigiPileupCorrHistogramMaker() {

  for(std::map<unsigned int,std::string>::const_iterator lab=m_labels.begin();lab!=m_labels.end();lab++) {

    const unsigned int i = lab->first; const std::string slab = lab->second;

    delete m_subdirs[i];
  }

}



void DigiPileupCorrHistogramMaker::book(const std::string dirname, const std::map<unsigned int, std::string>& labels) {

  m_labels = labels;
  book(dirname);

}

void DigiPileupCorrHistogramMaker::book(const std::string dirname) {

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
      sprintf(name,"n%sdigivsmclumi",slab.c_str());
      sprintf(title,"%s %s multiplicity ave pileup interactions",slab.c_str(),m_hitname.c_str());
      m_nmultvsmclumi[i] = m_subdirs[i]->make<TH2F>(name,title,200,0.,50.,m_nbins,0.,m_binmax[i]/(m_scalefact*m_nbins)*m_nbins);
      m_nmultvsmclumi[i]->GetXaxis()->SetTitle("Average Pileup Interactions"); m_nmultvsmclumi[i]->GetYaxis()->SetTitle("Number of Hits");
      sprintf(name,"n%sdigivsmclumiprof",slab.c_str());
      m_nmultvsmclumiprof[i] = m_subdirs[i]->make<TProfile>(name,title,200,0.,50.);
      m_nmultvsmclumiprof[i]->GetXaxis()->SetTitle("Average Pileup Interactions");    m_nmultvsmclumiprof[i]->GetYaxis()->SetTitle("Number of Hits");

      sprintf(name,"n%sdigivsmcnvtx",slab.c_str());
      sprintf(title,"%s %s multiplicity vs pileup interactions",slab.c_str(),m_hitname.c_str());
      m_nmultvsmcnvtx[i] = m_subdirs[i]->make<TH2F>(name,title,60,-0.5,59.5,m_nbins,0.,m_binmax[i]/(m_scalefact*m_nbins)*m_nbins);
      m_nmultvsmcnvtx[i]->GetXaxis()->SetTitle("Pileup Interactions"); m_nmultvsmcnvtx[i]->GetYaxis()->SetTitle("Number of Hits");
      sprintf(name,"n%sdigivsmcnvtxprof",slab.c_str());
      m_nmultvsmcnvtxprof[i] = m_subdirs[i]->make<TProfile>(name,title,60,-0.5,59.5);
      m_nmultvsmcnvtxprof[i]->GetXaxis()->SetTitle("Pileup Interactions");    m_nmultvsmcnvtxprof[i]->GetYaxis()->SetTitle("Number of Hits");

    }

  }


}

void DigiPileupCorrHistogramMaker::beginRun(const unsigned int nrun) {


}

void DigiPileupCorrHistogramMaker::fill(const edm::Event& iEvent, const std::map<unsigned int,int>& ndigi) {

  edm::Handle<std::vector<PileupSummaryInfo> > pileupinfos;
  iEvent.getByToken(m_pileupcollectionToken,pileupinfos);

  // look for the intime PileupSummaryInfo

  std::vector<PileupSummaryInfo>::const_iterator pileupinfo;

  for(pileupinfo = pileupinfos->begin(); pileupinfo != pileupinfos->end() ; ++pileupinfo) {

    if(pileupinfo->getBunchCrossing()==0) break;

  }

  if(pileupinfo->getBunchCrossing()!=0) {

    edm::LogError("NoInTimePileUpInfo") << "Cannot find the in-time pileup info " << pileupinfo->getBunchCrossing();

  }
  else {

    int npileup = pileupinfo->getPU_NumInteractions();

    if(m_useVisibleVertices) npileup = pileupinfo->getPU_zpositions().size();

    for(std::map<unsigned int,int>::const_iterator digi=ndigi.begin();digi!=ndigi.end();digi++) {
      if(m_labels.find(digi->first) != m_labels.end()) {
	const unsigned int i=digi->first;
	m_nmultvsmcnvtx[i]->Fill(npileup,digi->second);
	m_nmultvsmcnvtxprof[i]->Fill(npileup,digi->second);
	m_nmultvsmclumi[i]->Fill(pileupinfo->getTrueNumInteractions(),digi->second);
	m_nmultvsmclumiprof[i]->Fill(pileupinfo->getTrueNumInteractions(),digi->second);
      }
    }
  }
}


