#include "DPGAnalysis/SiStripTools/interface/TSOSHistogramMaker.h"
#include <iostream>
#include "TH1F.h"
#include "TH2F.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

TSOSHistogramMaker::TSOSHistogramMaker(): 
  m_2dhistos(false), 
  m_detsels(), m_selnames(), m_seltitles(), m_histocluslenangle(), 
  m_tsosy(), m_tsosx(), m_tsosxy(), 
  m_ttrhy(), m_ttrhx(), m_ttrhxy(), 
  m_tsosdy(), m_tsosdx(), m_tsosdxdy() 
{}

TSOSHistogramMaker::TSOSHistogramMaker(const edm::ParameterSet& iConfig): 
  m_2dhistos(iConfig.getUntrackedParameter<bool>("wanted2DHistos",false)),
  m_detsels(), m_selnames(), m_seltitles(), m_histocluslenangle(), 
  m_tsosy(), m_tsosx(), m_tsosxy(), 
  m_ttrhy(), m_ttrhx(), m_ttrhxy(), 
  m_tsosdy(), m_tsosdx(), m_tsosdxdy() 
{
  
  edm::Service<TFileService> tfserv;

  std::vector<edm::ParameterSet> wantedsubds(iConfig.getParameter<std::vector<edm::ParameterSet> >("wantedSubDets"));
					     
  std::cout << "selections found: " << wantedsubds.size() << std::endl;

  for(std::vector<edm::ParameterSet>::iterator ps=wantedsubds.begin();ps!=wantedsubds.end();++ps) {
    m_selnames.push_back(ps->getParameter<std::string>("name"));
    m_seltitles.push_back(ps->getParameter<std::string>("title"));
    m_detsels.push_back(DetIdSelector(ps->getParameter<std::vector<std::string> >("selection")));

    TFileDirectory subdir = tfserv->mkdir(ps->getParameter<std::string>("name"));

    std::string name = "tsosy_" + ps->getParameter<std::string>("name");
    std::string title = "TSOS y " + ps->getParameter<std::string>("title");
    m_tsosy.push_back(subdir.make<TH1F>(name.c_str(),title.c_str(),200,-20.,20.));
    name = "tsosx_" + ps->getParameter<std::string>("name");
    title = "TSOS x " + ps->getParameter<std::string>("title");
    m_tsosx.push_back(subdir.make<TH1F>(name.c_str(),title.c_str(),200,-20.,20.));
    if(m_2dhistos) {
      name = "tsosxy_" + ps->getParameter<std::string>("name");
      title = "TSOS y vs x " + ps->getParameter<std::string>("title");
      m_tsosxy.push_back(subdir.make<TH2F>(name.c_str(),title.c_str(),200,-20.,20.,200,-20.,20.));
    }

    name = "tsosprojx_" + ps->getParameter<std::string>("name");
    title = "TSOS x projection " + ps->getParameter<std::string>("title");
    m_tsosprojx.push_back(subdir.make<TH1F>(name.c_str(),title.c_str(),400,-2.,2.));
    name = "tsosprojy_" + ps->getParameter<std::string>("name");
    title = "TSOS y projection " + ps->getParameter<std::string>("title");
    m_tsosprojy.push_back(subdir.make<TH1F>(name.c_str(),title.c_str(),400,-2.,2.));

    name = "ttrhy_" + ps->getParameter<std::string>("name");
    title = "TT RecHit y " + ps->getParameter<std::string>("title");
    m_ttrhy.push_back(subdir.make<TH1F>(name.c_str(),title.c_str(),200,-20.,20.));
    name = "ttrhx_" + ps->getParameter<std::string>("name");
    title = "TT RecHit x " + ps->getParameter<std::string>("title");
    m_ttrhx.push_back(subdir.make<TH1F>(name.c_str(),title.c_str(),200,-20.,20.));
    if(m_2dhistos) {
      name = "ttrhxy_" + ps->getParameter<std::string>("name");
      title = "TT RecHit y vs x  " + ps->getParameter<std::string>("title");
      m_ttrhxy.push_back(subdir.make<TH2F>(name.c_str(),title.c_str(),200,-20.,20.,200,-20.,20.));
    }

    name = "tsosdy_" + ps->getParameter<std::string>("name");
    title = "TSOS-TTRH y " + ps->getParameter<std::string>("title");
    m_tsosdy.push_back(subdir.make<TH1F>(name.c_str(),title.c_str(),200,-5.,5.));
    name = "tsosdx_" + ps->getParameter<std::string>("name");
    title = "TSOS-TTRH x " + ps->getParameter<std::string>("title");
    m_tsosdx.push_back(subdir.make<TH1F>(name.c_str(),title.c_str(),200,-0.1,0.1));
    if(m_2dhistos) {
      name = "tsosdxdy_" + ps->getParameter<std::string>("name");
      title = "TSOS-TTRH dy vs dy " + ps->getParameter<std::string>("title");
      m_tsosdxdy.push_back(subdir.make<TH2F>(name.c_str(),title.c_str(),200,-0.1,0.1,200,-5.,5.));
    }

    name = "cluslenangle_" + ps->getParameter<std::string>("name");
    title = "Cluster Length vs Track Angle " + ps->getParameter<std::string>("title");
    m_histocluslenangle.push_back(subdir.make<TH2F>(name.c_str(),title.c_str(),200,-1.,1.,40,-0.5,39.5));
		       
  }
}

void TSOSHistogramMaker::fill(const TrajectoryStateOnSurface& tsos, TransientTrackingRecHit::ConstRecHitPointer hit) const {
  
  if(hit==0 || !hit->isValid()) return;
  
  for(unsigned int i=0; i<m_detsels.size() ; ++i) {
    
    if(m_detsels[i].isSelected(hit->geographicalId())) {
      
      m_tsosy[i]->Fill(tsos.localPosition().y());
      m_tsosx[i]->Fill(tsos.localPosition().x());
      if(m_2dhistos)       m_tsosxy[i]->Fill(tsos.localPosition().x(),tsos.localPosition().y());
      m_ttrhy[i]->Fill(hit->localPosition().y());
      m_ttrhx[i]->Fill(hit->localPosition().x());
      if(m_2dhistos)       m_ttrhxy[i]->Fill(hit->localPosition().x(),hit->localPosition().y());
      m_tsosdy[i]->Fill(tsos.localPosition().y()-hit->localPosition().y());
      m_tsosdx[i]->Fill(tsos.localPosition().x()-hit->localPosition().x());
      if(m_2dhistos)       m_tsosdxdy[i]->Fill(tsos.localPosition().x()-hit->localPosition().x(),tsos.localPosition().y()-hit->localPosition().y());

      if(tsos.localDirection().z() != 0) {
	m_tsosprojx[i]->Fill(tsos.localDirection().x()/tsos.localDirection().z());
	m_tsosprojy[i]->Fill(tsos.localDirection().y()/tsos.localDirection().z());
      }

    }
    
  }
  
  
}



