/*
 * \file BeamConditionsMonitor.cc
 * \author Geng-yuan Jeng/UC Riverside
 *         Francisco Yumiceva/FNAL
 *
 */

#include "DQM/BeamMonitor/plugins/BeamConditionsMonitor.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "CondFormats/DataRecord/interface/BeamSpotObjectsRcd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include <numeric>
#include <cmath>
#include <TMath.h>
#include <iostream>
#include "TStyle.h"

using namespace std;
using namespace edm;

//
// constructors and destructor
//
BeamConditionsMonitor::BeamConditionsMonitor( const ParameterSet& ps ):
  bsSrc_{ps.getUntrackedParameter<InputTag>("beamSpot")}

  {

  monitorName_    = ps.getUntrackedParameter<string>("monitorName","YourSubsystemName");
  
  if (!monitorName_.empty() ) monitorName_ = monitorName_+"/" ;
}


//--------------------------------------------------------
void BeamConditionsMonitor::bookHistograms(DQMStore::ConcurrentBooker& i, const edm::Run& r, const edm::EventSetup& c, 
                                           beamcond::RunCache& cache) const {
  
  // book some histograms here
  // create and cd into new folder
  i.setCurrentFolder(monitorName_+"Conditions");
  
  cache.h_x0_lumi = i.book1D("x0_lumi_cond","x coordinate of beam spot vs lumi (Cond)",10,0,10);
  cache.h_x0_lumi.setAxisTitle("Lumisection",1);
  cache.h_x0_lumi.setAxisTitle("x_{0} (cm)",2);
  cache.h_x0_lumi.setOption("E1");

  cache.h_y0_lumi = i.book1D("y0_lumi_cond","y coordinate of beam spot vs lumi (Cond)",10,0,10);
  cache.h_y0_lumi.setAxisTitle("Lumisection",1);
  cache.h_y0_lumi.setAxisTitle("y_{0} (cm)",2);
  cache.h_y0_lumi.setOption("E1");
  
}


std::shared_ptr<void> 
BeamConditionsMonitor::globalBeginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                                                  const edm::EventSetup& c) const {
  ESHandle< BeamSpotObjects > beamhandle;
  c.get<BeamSpotObjectsRcd>().get(beamhandle);
  auto const& condBeamSpot = *beamhandle;

  auto cache = runCache(lumiSeg.getRun().index() );
  LogInfo("BeamConditions") << "[BeamConditionsMonitor]:" << condBeamSpot << endl;
  cache->h_x0_lumi.shiftFillLast( condBeamSpot.GetX(), condBeamSpot.GetXError(), 1 );
  cache->h_y0_lumi.shiftFillLast( condBeamSpot.GetY(), condBeamSpot.GetYError(), 1 );

  return std::shared_ptr<void>{};
}

// ----------------------------------------------------------
void BeamConditionsMonitor::dqmAnalyze(const Event& iEvent, const EventSetup& iSetup, 
                                       beamcond::RunCache const&) const {

}


//--------------------------------------------------------
void BeamConditionsMonitor::globalEndLuminosityBlock(const LuminosityBlock& lumiSeg, 
                                                     const EventSetup& iSetup) const {


}

DEFINE_FWK_MODULE(BeamConditionsMonitor);
