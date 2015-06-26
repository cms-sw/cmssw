#include "DPGAnalysis/SiStripTools/interface/MultiplicityCorrelatorHistogramMaker.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TH2F.h"
#include "TH1F.h"
#include <cmath>

MultiplicityCorrelatorHistogramMaker::MultiplicityCorrelatorHistogramMaker(edm::ConsumesCollector&& iC):
  m_rhm(iC, false), m_fhm(iC, true), m_runHisto(false), m_runHistoBXProfile(false), m_runHistoBX(false), m_runHisto2D(false), m_runHistoProfileBX(false),
  m_scfact(1.), m_yvsxmult(0),
  m_atanyoverx(0), m_atanyoverxrun(0), m_atanyoverxvsbxrun(0), m_atanyoverxvsbxrun2D(0),
  m_yvsxmultrun(0), m_yvsxmultprofvsbxrun(0), m_xvsymultprofvsbxrun(0)
{}

MultiplicityCorrelatorHistogramMaker::MultiplicityCorrelatorHistogramMaker(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& iC):
  m_rhm(iC, false), m_fhm(iC, true),
  m_runHisto(iConfig.getParameter<bool>("runHisto")),
  m_runHistoBXProfile(iConfig.getParameter<bool>("runHistoBXProfile")),
  m_runHistoBX(iConfig.getParameter<bool>("runHistoBX")),
  m_runHisto2D(iConfig.getParameter<bool>("runHisto2D")),
  m_runHistoProfileBX(iConfig.getUntrackedParameter<bool>("runHistoProfileBX",false)),
  m_scfact(iConfig.getUntrackedParameter<double>("scaleFactor",1.)),
  m_atanyoverxrun(0), m_atanyoverxvsbxrun(0), m_atanyoverxvsbxrun2D(0), m_yvsxmultrun(0),
  m_yvsxmultprofvsbxrun(0), m_xvsymultprofvsbxrun(0)
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

  if(m_runHisto && m_runHistoProfileBX) {
    sprintf(hname,"%sVs%sprofvsbx",
	    iConfig.getParameter<std::string>("yDetLabel").c_str(),
	    iConfig.getParameter<std::string>("xDetLabel").c_str());
    sprintf(htitle,"%s Vs %s multiplicity vs BX",
	    iConfig.getParameter<std::string>("yDetLabel").c_str(),
	    iConfig.getParameter<std::string>("xDetLabel").c_str());
    m_yvsxmultprofvsbxrun = m_fhm.makeTProfile2D(hname,htitle,
						 3564,-0.5,3564-0.5,
						 iConfig.getParameter<unsigned int>("xBins"),0.,iConfig.getParameter<double>("xMax"));
    sprintf(hname,"%sVs%sprofvsbx",
	    iConfig.getParameter<std::string>("xDetLabel").c_str(),
	    iConfig.getParameter<std::string>("yDetLabel").c_str());
    sprintf(htitle,"%s Vs %s multiplicity vs BX",
	    iConfig.getParameter<std::string>("xDetLabel").c_str(),
	    iConfig.getParameter<std::string>("yDetLabel").c_str());
    m_xvsymultprofvsbxrun = m_fhm.makeTProfile2D(hname,htitle,
						 3564,-0.5,3564-0.5,
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
    sprintf(hname,"%sOver%srun",
	    iConfig.getParameter<std::string>("yDetLabel").c_str(),
	    iConfig.getParameter<std::string>("xDetLabel").c_str());
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
      m_atanyoverxvsbxrun2D = m_fhm.makeTH2F(hname,htitle,3564,-0.5,3564-0.5,
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
      m_atanyoverxvsbxrun = m_fhm.makeTProfile(hname,htitle,3564,-0.5,3564-0.5);
    }
  }

}


MultiplicityCorrelatorHistogramMaker::~MultiplicityCorrelatorHistogramMaker() { }

void MultiplicityCorrelatorHistogramMaker::beginRun(const edm::Run& iRun) {

  m_rhm.beginRun(iRun);
  m_fhm.beginRun(iRun);

}

void MultiplicityCorrelatorHistogramMaker::fill(const edm::Event& iEvent, const int xmult, const int ymult) {



  const int bx = iEvent.bunchCrossing();

  if(m_yvsxmult) m_yvsxmult->Fill(xmult,ymult);
  if(m_atanyoverx) m_atanyoverx->Fill(atan2(ymult*m_scfact,xmult));

  if(m_yvsxmultrun && *m_yvsxmultrun) (*m_yvsxmultrun)->Fill(xmult,ymult);
  if(m_atanyoverxrun && *m_atanyoverxrun) (*m_atanyoverxrun)->Fill(atan2(ymult*m_scfact,xmult));
  if(m_atanyoverxvsbxrun && *m_atanyoverxvsbxrun) (*m_atanyoverxvsbxrun)->Fill(bx%3564,atan2(ymult*m_scfact,xmult));
  if(m_atanyoverxvsbxrun2D && *m_atanyoverxvsbxrun2D) (*m_atanyoverxvsbxrun2D)->Fill(bx%3564,atan2(ymult*m_scfact,xmult));

  if(m_yvsxmultprofvsbxrun && *m_yvsxmultprofvsbxrun) (*m_yvsxmultprofvsbxrun)->Fill(bx%3564,xmult,ymult);
  if(m_xvsymultprofvsbxrun && *m_xvsymultprofvsbxrun) (*m_xvsymultprofvsbxrun)->Fill(bx%3564,ymult,xmult);
}

