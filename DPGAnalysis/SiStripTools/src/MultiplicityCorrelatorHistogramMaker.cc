#include "DPGAnalysis/SiStripTools/interface/MultiplicityCorrelatorHistogramMaker.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TH2F.h"
#include "TH1F.h"
#include <cmath>

MultiplicityCorrelatorHistogramMaker::MultiplicityCorrelatorHistogramMaker():
  m_rhm(), m_runHisto(false), m_runHistoBXProfile(false), m_runHistoBX(false), m_runHisto2D(false),
  m_scfact(1.), m_yvsxmult(0), m_atanyoverx(0), m_atanyoverxrun(0), m_atanyoverxvsbxrun(0), m_atanyoverxvsbxrun2D(0), m_yvsxmultrun(0) {}

MultiplicityCorrelatorHistogramMaker::MultiplicityCorrelatorHistogramMaker(const edm::ParameterSet& iConfig):
  m_rhm(),
  m_runHisto(iConfig.getParameter<bool>("runHisto")),
  m_runHistoBXProfile(iConfig.getParameter<bool>("runHistoBXProfile")),
  m_runHistoBX(iConfig.getParameter<bool>("runHistoBX")),
  m_runHisto2D(iConfig.getParameter<bool>("runHisto2D")),
  m_scfact(iConfig.getUntrackedParameter<double>("scaleFactor",1.)),
  m_atanyoverxrun(0), m_atanyoverxvsbxrun(0), m_atanyoverxvsbxrun2D(0), m_yvsxmultrun(0) 
 { 

  edm::Service<TFileService> tfserv;

  char hname[300];
  sprintf(hname,"%sVs%s",
	  iConfig.getParameter<std::string>("yDetLabel").c_str(),
	  iConfig.getParameter<std::string>("xDetLabel").c_str());
  char htitle[300];
  sprintf(htitle,"%s Vs %s multiplicity",
	  iConfig.getParameter<std::string>("yDetLabel").c_str(),
	  iConfig.getParameter<std::string>("xDetLabel").c_str());

  m_yvsxmult = tfserv->make<TH2F>(hname,htitle,
		     iConfig.getParameter<unsigned int>("xBins"),0.,iConfig.getParameter<double>("xMax"),
		     iConfig.getParameter<unsigned int>("yBins"),0.,iConfig.getParameter<double>("yMax"));

  if(m_runHisto && m_runHisto2D) {
    m_yvsxmultrun = m_rhm.makeTH2F(hname,htitle,
		     iConfig.getParameter<unsigned int>("xBins"),0.,iConfig.getParameter<double>("xMax"),
		     iConfig.getParameter<unsigned int>("yBins"),0.,iConfig.getParameter<double>("yMax"));
  }

  sprintf(hname,"%sOver%s",
	  iConfig.getParameter<std::string>("yDetLabel").c_str(),
	  iConfig.getParameter<std::string>("xDetLabel").c_str());
  sprintf(htitle,"atan (%4.2f*%s / %s multiplicity ratio)",
	  m_scfact,	  
	  iConfig.getParameter<std::string>("yDetLabel").c_str(),
	  iConfig.getParameter<std::string>("xDetLabel").c_str()
	  );

  m_atanyoverx = tfserv->make<TH1F>(hname,htitle,
				   iConfig.getParameter<unsigned int>("rBins"),0.,1.6);
    
  if(m_runHisto) {
    m_atanyoverxrun = m_rhm.makeTH1F(hname,htitle,
				     iConfig.getParameter<unsigned int>("rBins"),0.,1.6);
    if(m_runHistoBX) {

      sprintf(hname,"%sOver%svsbx2D",
	      iConfig.getParameter<std::string>("yDetLabel").c_str(),
	      iConfig.getParameter<std::string>("xDetLabel").c_str());
      sprintf(htitle,"atan (%4.2f*%s / %s multiplicity ratio)",
	      m_scfact,	      
	      iConfig.getParameter<std::string>("yDetLabel").c_str(),
	      iConfig.getParameter<std::string>("xDetLabel").c_str()
	      );
      m_atanyoverxvsbxrun2D = m_rhm.makeTH2F(hname,htitle,3564,-0.5,3563.5,
					   iConfig.getParameter<unsigned int>("rBins"),0.,1.6);
    }
    if(m_runHistoBXProfile) {
      sprintf(hname,"%sOver%svsbx",
	      iConfig.getParameter<std::string>("yDetLabel").c_str(),
	      iConfig.getParameter<std::string>("xDetLabel").c_str());
      sprintf(htitle,"atan (%4.2f*%s / %s multiplicity ratio)",
	      m_scfact,
	      iConfig.getParameter<std::string>("yDetLabel").c_str(),
	      iConfig.getParameter<std::string>("xDetLabel").c_str()
	      );
      m_atanyoverxvsbxrun = m_rhm.makeTProfile(hname,htitle,3564,-0.5,3563.5);
    }
  }

}


MultiplicityCorrelatorHistogramMaker::~MultiplicityCorrelatorHistogramMaker() { }

void MultiplicityCorrelatorHistogramMaker::beginRun(const unsigned int nrun) {

  m_rhm.beginRun(nrun);

}

void MultiplicityCorrelatorHistogramMaker::fill(const int xmult, const int ymult, const int bx) {
  
  if(m_yvsxmult) m_yvsxmult->Fill(xmult,ymult);
  if(m_atanyoverx) m_atanyoverx->Fill(atan2(ymult*m_scfact,xmult));

  if(m_yvsxmultrun && *m_yvsxmultrun) (*m_yvsxmultrun)->Fill(xmult,ymult);
  if(m_atanyoverxrun && *m_atanyoverxrun) (*m_atanyoverxrun)->Fill(atan2(ymult*m_scfact,xmult));
  if(m_atanyoverxvsbxrun && *m_atanyoverxvsbxrun) (*m_atanyoverxvsbxrun)->Fill(bx,atan2(ymult*m_scfact,xmult));
  if(m_atanyoverxvsbxrun2D && *m_atanyoverxvsbxrun2D) (*m_atanyoverxvsbxrun2D)->Fill(bx,atan2(ymult*m_scfact,xmult));

}

