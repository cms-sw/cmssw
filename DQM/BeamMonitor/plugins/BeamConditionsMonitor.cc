/*
 * \file BeamConditionsMonitor.cc
 * \author Geng-yuan Jeng/UC Riverside
 *         Francisco Yumiceva/FNAL
 * $Date: 2009/08/05 14:45:09 $
 * $Revision: 1.10 $
 *
 */

#include "DQM/BeamMonitor/interface/BeamConditionsMonitor.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include <numeric>
#include <math.h>
#include <TMath.h>
#include <iostream>
#include "TStyle.h"

using namespace std;
using namespace edm;

//
// constructors and destructor
//
BeamConditionsMonitor::BeamConditionsMonitor( const ParameterSet& ps ) :
  countEvt_(0),countLumi_(0)
{
  parameters_     = ps;
  monitorName_    = parameters_.getUntrackedParameter<string>("monitorName","YourSubsystemName");
  bsSrc_          = parameters_.getUntrackedParameter<string>("beamSpot","offlineBeamSpot");
  debug_          = parameters_.getUntrackedParameter<bool>("Debug");

  dbe_            = Service<DQMStore>().operator->();
  
  if (monitorName_ != "" ) monitorName_ = monitorName_+"/" ;
  
}


BeamConditionsMonitor::~BeamConditionsMonitor()
{
}


//--------------------------------------------------------
void BeamConditionsMonitor::beginJob(const EventSetup& context){
  
  // book some histograms here
  // create and cd into new folder
  dbe_->setCurrentFolder(monitorName_+"Conditions");
  
  h_x0_lumi = dbe_->book1D("x0_lumi_cond","x_{0} vs lumi (Cond)",10,0,10);
  h_x0_lumi->setAxisTitle("Lumisection",1);
  h_x0_lumi->setAxisTitle("x_{0} (cm)",2);
  h_x0_lumi->getTH1()->SetOption("E1");

  h_y0_lumi = dbe_->book1D("y0_lumi_cond","y_{0} vs lumi (Cond)",10,0,10);
  h_y0_lumi->setAxisTitle("Lumisection",1);
  h_y0_lumi->setAxisTitle("y_{0} (cm)",2);
  h_y0_lumi->getTH1()->SetOption("E1");
  
}

//--------------------------------------------------------
void BeamConditionsMonitor::beginRun(const edm::Run& r, const EventSetup& context) {
  
}

//--------------------------------------------------------
void BeamConditionsMonitor::beginLuminosityBlock(const LuminosityBlock& lumiSeg, 
					     const EventSetup& context) {
  countLumi_++;
}

// ----------------------------------------------------------
void BeamConditionsMonitor::analyze(const Event& iEvent, 
				const EventSetup& iSetup )
{  
  countEvt_++;
  Handle<reco::BeamSpot> recoBeamSpotHandle;
  iEvent.getByLabel(bsSrc_,recoBeamSpotHandle);
  theBS = recoBeamSpotHandle.product();

}


//--------------------------------------------------------
void BeamConditionsMonitor::endLuminosityBlock(const LuminosityBlock& lumiSeg, 
					   const EventSetup& iSetup) {

  h_x0_lumi->ShiftFillLast( theBS->x0(), theBS->x0Error(), 1 );
  h_y0_lumi->ShiftFillLast( theBS->y0(), theBS->y0Error(), 1 );

}
//--------------------------------------------------------
void BeamConditionsMonitor::endRun(const Run& r, const EventSetup& context){
  
  
}
//--------------------------------------------------------
void BeamConditionsMonitor::endJob(){

}

DEFINE_FWK_MODULE(BeamConditionsMonitor);
