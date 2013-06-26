#include "DPGAnalysis/SiStripTools/interface/DigiInvestigatorHistogramMaker.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TProfile.h"
#include "TH1F.h"

#include "DPGAnalysis/SiStripTools/interface/SiStripTKNumbers.h"


DigiInvestigatorHistogramMaker::DigiInvestigatorHistogramMaker():
  _hitname(), _nbins(500), m_maxLS(100), m_LSfrac(4), _scalefact(), _runHisto(true), _binmax(), _labels(), _nmultvsorbrun(), _nmult() { }

DigiInvestigatorHistogramMaker::DigiInvestigatorHistogramMaker(const edm::ParameterSet& iConfig):
  _hitname(iConfig.getUntrackedParameter<std::string>("hitName","digi")),
  _nbins(iConfig.getUntrackedParameter<int>("numberOfBins",500)),
  m_maxLS(iConfig.getUntrackedParameter<unsigned int>("maxLSBeforeRebin",100)),
  m_LSfrac(iConfig.getUntrackedParameter<unsigned int>("startingLSFraction",4)),
  _scalefact(iConfig.getUntrackedParameter<int>("scaleFactor",5)),
  _runHisto(iConfig.getUntrackedParameter<bool>("runHisto",true)),
  _labels(), _rhm(), _nmultvsorbrun(), _nmult(), _subdirs() 
{ 

  std::vector<edm::ParameterSet> 
    wantedsubds(iConfig.getUntrackedParameter<std::vector<edm::ParameterSet> >("wantedSubDets",std::vector<edm::ParameterSet>()));
  
  for(std::vector<edm::ParameterSet>::iterator ps=wantedsubds.begin();ps!=wantedsubds.end();++ps) {
    _labels[ps->getParameter<unsigned int>("detSelection")] = ps->getParameter<std::string>("detLabel");
    _binmax[ps->getParameter<unsigned int>("detSelection")] = ps->getParameter<int>("binMax");
  }
  

}


DigiInvestigatorHistogramMaker::~DigiInvestigatorHistogramMaker() {

  for(std::map<unsigned int,std::string>::const_iterator lab=_labels.begin();lab!=_labels.end();lab++) {
    
    const unsigned int i = lab->first; const std::string slab = lab->second;
    
    delete _subdirs[i];
  }
  
}



void DigiInvestigatorHistogramMaker::book(const std::string dirname, const std::map<unsigned int, std::string>& labels) {

  _labels = labels;
  book(dirname);

}

void DigiInvestigatorHistogramMaker::book(const std::string dirname) {

  edm::Service<TFileService> tfserv;
  TFileDirectory subev = tfserv->mkdir(dirname);

  SiStripTKNumbers trnumb;
  
  edm::LogInfo("NumberOfBins") << "Number of Bins: " << _nbins;
  edm::LogInfo("NumberOfMaxLS") << "Max number of LS before rebinning: " << m_maxLS;
  edm::LogInfo("StartingLSFrac") << "Fraction of LS in one bin before rebinning: " << m_LSfrac;
  edm::LogInfo("ScaleFactors") << "x-axis range scale factor: " << _scalefact;
  edm::LogInfo("BinMaxValue") << "Setting bin max values";

  for(std::map<unsigned int,std::string>::const_iterator lab=_labels.begin();lab!=_labels.end();lab++) {
    
    const unsigned int i = lab->first; const std::string slab = lab->second;
    
    if(_binmax.find(i)==_binmax.end()) {
      edm::LogVerbatim("NotConfiguredBinMax") << "Bin max for " << lab->second 
					      << " not configured: " << trnumb.nstrips(i) << " used";
      _binmax[i] = trnumb.nstrips(i);
    }
 
    edm::LogVerbatim("BinMaxValue") << "Bin max for " << lab->second << " is " << _binmax[i];

  }

  for(std::map<unsigned int,std::string>::const_iterator lab=_labels.begin();lab!=_labels.end();++lab) {

    const int i = lab->first; const std::string slab = lab->second;

    char name[200];
    char title[500];

    _subdirs[i] = new TFileDirectory(subev.mkdir(slab.c_str()));

    if(_subdirs[i]) {
      sprintf(name,"n%sdigi",slab.c_str());
      sprintf(title,"%s %s multiplicity",slab.c_str(),_hitname.c_str());
      _nmult[i] = _subdirs[i]->make<TH1F>(name,title,_nbins,0.,(1+_binmax[i]/(_scalefact*_nbins))*_nbins);
      _nmult[i]->GetXaxis()->SetTitle("Number of Hits");    _nmult[i]->GetYaxis()->SetTitle("Events");
      
      if(_runHisto) {
	sprintf(name,"n%sdigivsorbrun",slab.c_str());
	sprintf(title,"%s %s mean multiplicity vs orbit",slab.c_str(),_hitname.c_str());
	_nmultvsorbrun[i] = _rhm.makeTProfile(name,title,m_LSfrac*m_maxLS,0,m_maxLS*262144);
      }

    }

  }


}

void DigiInvestigatorHistogramMaker::beginRun(const unsigned int nrun) {

  //  char runname[100];
  //  sprintf(runname,"run_%d",nrun);

  edm::Service<TFileService> tfserv;

  //  currdir = &(*tfserv);
  //  _rhm.beginRun(nrun,*currdir);

  _rhm.beginRun(nrun,*tfserv);


  for(std::map<unsigned int,std::string>::const_iterator lab=_labels.begin();lab!=_labels.end();++lab) {

    const int i = lab->first; const std::string slab = lab->second;

    //    char name[200];
    //    char title[500];

    //    TFileDirectory subd =_subdirs[i]->mkdir(runname);

    //    sprintf(name,"n%sdigivsorbrun",slab.c_str());
    //    sprintf(title,"%s %s mean multiplicity vs orbit",slab.c_str(),_hitname.c_str());
    //    _nmultvsorbrun[i] = subd.make<TProfile>(name,title,_norbbin,0.5,11223*_norbbin+0.5);
    if(_runHisto) {
      (*_nmultvsorbrun[i])->GetXaxis()->SetTitle("time [orbit#]");    (*_nmultvsorbrun[i])->GetYaxis()->SetTitle("Hits");
      (*_nmultvsorbrun[i])->SetBit(TH1::kCanRebin);
    }
  }


}

void DigiInvestigatorHistogramMaker::fill(const unsigned int orbit, const std::map<unsigned int,int>& ndigi) {
  
  for(std::map<unsigned int,int>::const_iterator digi=ndigi.begin();digi!=ndigi.end();digi++) {
    
    if(_labels.find(digi->first) != _labels.end()) {
 
      const unsigned int i=digi->first;
      
      _nmult[i]->Fill(digi->second);
      if(_runHisto) {
	if(_nmultvsorbrun[i] && *_nmultvsorbrun[i]) (*_nmultvsorbrun[i])->Fill(orbit,digi->second);
      }
    }
    
  }
}

