/*
 * \file DTTnPEfficiencyTask.cc
 *
 * \author L. Lunerti - INFN Bologna
 *
 */

#include "DQM/DTMonitorModule/src/DTTnPEfficiencyTask.h"

// Framework
#include "FWCore/Framework/interface/EventSetup.h"

// Geometry
#include "DataFormats/GeometryVector/interface/Pi.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"

//Root
#include "TH1.h"
#include "TAxis.h"

#include <sstream>
#include <iostream>
#include <fstream>

DTTnPEfficiencyTask::DTTnPEfficiencyTask(const edm::ParameterSet& config) : 
  m_nEvents(0),
  m_muToken(consumes<reco::MuonCollection>(config.getUntrackedParameter<edm::InputTag>("inputTagMuons"))),
  m_detailedAnalysis(config.getUntrackedParameter<bool>("detailedAnalysis")),
  m_selector(config.getUntrackedParameter<std::string>("probeCut")),
  m_borderCut(-10.)
{

  LogTrace("DTDQM|DTMonitorModule|DTTnPEfficiencyTask")
    << "[DTTnPEfficiencyTask]: Constructor" << std::endl;

}

DTTnPEfficiencyTask::~DTTnPEfficiencyTask() 
{

  LogTrace("DTDQM|DTMonitorModule|DTTnPEfficiencyTask")
    << "[DTTnPEfficiencyTask]: analyzed " << m_nEvents << " events" << std::endl;

}

void DTTnPEfficiencyTask::dqmBeginRun(const edm::Run& run, const edm::EventSetup& context) 
{

}

void DTTnPEfficiencyTask::bookHistograms(DQMStore::IBooker& iBooker, edm::Run const& run,
					 edm::EventSetup const& context) 
{

  LogTrace("DTDQM|DTMonitorModule|DTTnPEfficiencyTask") 
    << "[DTTnPEfficiencyTask]: bookHistograms" << std::endl;

  for (int wheel = -2; wheel <= 2; ++wheel) 
    {
      if (m_detailedAnalysis) 
	{
	  std::string baseDir = topFolder() + "/detailed/";
	  iBooker.setCurrentFolder(baseDir);

	  LogTrace("DTDQM|DTMonitorModule|DTTnPEfficiencyTask")
	    << "[DTTnPEfficiencyTask]: booking histos in " << baseDir << std::endl;

	  MonitorElement* me = iBooker.book1D("muonPt", "muonPt", 250, 0., 500.);

	  m_histos["muonPt"] = me;
	}
      
      bookWheelHistos(iBooker, wheel, "Task");
    }

}

void DTTnPEfficiencyTask::analyze(const edm::Event& event, const edm::EventSetup& context) 
{
  ++m_nEvents;

  edm::Handle<reco::MuonCollection> muons;
  event.getByToken(m_muToken, muons);

  for (const auto & muon : (*muons)) 
    {

      if (!m_selector(muon))
	continue;

      std::string hName = "muonPt"; 
      m_histos.find(hName)->second->Fill(muon.pt());

      for (const auto chambMatch : muon.matches() ) 
	{
	  
	  // look only in DTs
	  if (chambMatch.detector() != MuonSubdetId::DT)
	    continue;
 
	  if (chambMatch.edgeX < m_borderCut && 
	      chambMatch.edgeY < m_borderCut)
	    {
	      DTChamberId chId(chambMatch.id.rawId());

	      int wheel   = chId.wheel();
	      int sector  = chId.sector();
	      int station = chId.station();

	      hName = std::string("nEntriesPerCh_W") + std::to_string(wheel);  
	      m_histos.find(hName)->second->Fill(sector, station);
	    }
	  
	}
    
    }

}

void DTTnPEfficiencyTask::bookWheelHistos(DQMStore::IBooker& iBooker, 
					  int wheel, std::string folder) 
{

  auto baseDir = topFolder() + folder + "/";
  iBooker.setCurrentFolder(baseDir);

  LogTrace("DTDQM|DTMonitorModule|DTTnPEfficiencyTask")
    << "[DTTnPEfficiencyTask]: booking histos in " << baseDir << std::endl;

  auto hName = std::string("nEntriesPerCh_W") + std::to_string(wheel);    

  MonitorElement* me = iBooker.book2D(hName.c_str(), hName.c_str(), 14, 0.5, 14.5, 4, 0., 4.5);

  me->setBinLabel(1, "MB1", 2);
  me->setBinLabel(2, "MB2", 2);
  me->setBinLabel(3, "MB3", 2);
  me->setBinLabel(4, "MB4", 2);
  me->setAxisTitle("Sector", 1);

  m_histos[hName] = me;

}
