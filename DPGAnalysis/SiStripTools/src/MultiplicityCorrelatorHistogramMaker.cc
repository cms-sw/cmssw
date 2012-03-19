#include "DPGAnalysis/SiStripTools/interface/MultiplicityCorrelatorHistogramMaker.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TH2F.h"
#include "TH1F.h"
#include <cmath>

MultiplicityCorrelatorHistogramMaker::MultiplicityCorrelatorHistogramMaker():
  _scfact(1.), _yvsxmult(0), _atanyoverx(0) {}

MultiplicityCorrelatorHistogramMaker::MultiplicityCorrelatorHistogramMaker(const edm::ParameterSet& iConfig):
  _scfact(iConfig.getUntrackedParameter<double>("scaleFactor",1.))
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

  _yvsxmult = tfserv->make<TH2F>(hname,htitle,
		     iConfig.getParameter<unsigned int>("xBins"),0.,iConfig.getParameter<double>("xMax"),
		     iConfig.getParameter<unsigned int>("yBins"),0.,iConfig.getParameter<double>("yMax"));

  sprintf(hname,"%sOver%s",
	  iConfig.getParameter<std::string>("yDetLabel").c_str(),
	  iConfig.getParameter<std::string>("xDetLabel").c_str());
  sprintf(htitle,"atan (%s / %s multiplicity ratio): scale fact %f",
	  iConfig.getParameter<std::string>("yDetLabel").c_str(),
	  iConfig.getParameter<std::string>("xDetLabel").c_str(),
	  _scfact);

  _atanyoverx = tfserv->make<TH1F>(hname,htitle,
				   iConfig.getParameter<unsigned int>("rBins"),0.,1.6);
    

}


MultiplicityCorrelatorHistogramMaker::~MultiplicityCorrelatorHistogramMaker() { }



void MultiplicityCorrelatorHistogramMaker::fill(const int xmult, const int ymult) {
  
  if(_yvsxmult) _yvsxmult->Fill(xmult,ymult);
  if(_atanyoverx) _atanyoverx->Fill(atan2(ymult*_scfact,xmult));
}

