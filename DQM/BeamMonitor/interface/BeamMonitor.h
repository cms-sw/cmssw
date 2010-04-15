#ifndef BeamMonitor_H
#define BeamMonitor_H

/** \class BeamMonitor
 * *
 *  $Date: 2009/12/08 04:59:52 $
 *  $Revision: 1.10 $
 *  \author  Geng-yuan Jeng/UC Riverside
 *           Francisco Yumiceva/FNAL
 *   
 */
// C++
#include <string>
// CMS
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "RecoVertex/BeamSpotProducer/interface/BSTrkParameters.h"
#include "RecoVertex/BeamSpotProducer/interface/BeamFitter.h"
//#include "DataFormats/TrackReco/interface/TrackBase.h"

//
// class declaration
//

class BeamMonitor : public edm::EDAnalyzer {
 public:
  BeamMonitor( const edm::ParameterSet& );
  ~BeamMonitor();

 protected:
   
  // BeginJob
  void beginJob();

  // BeginRun
  void beginRun(const edm::Run& r, const edm::EventSetup& c);
  
  // Fake Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c) ;
  
  void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
			    const edm::EventSetup& context) ;
  
  // DQM Client Diagnostic
  void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
			  const edm::EventSetup& c);
  
  // EndRun
  void endRun(const edm::Run& r, const edm::EventSetup& c);
  
  // Endjob
  void endJob(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& c);
  
 private:
  
  edm::ParameterSet parameters_;
  std::string monitorName_;
  std::string bsSrc_; // beam spot
  edm::InputTag tracksLabel_;
  
  int fitNLumi_;
  int resetFitNLumi_;
  bool debug_;
  
  DQMStore* dbe_;
  BeamFitter * theBeamFitter;
  
  int countEvt_;       //counter
  int countLumi_;      //counter
  unsigned int nthBSTrk_;       //
  int nFitElements_;
  int nFits;
  double deltaSigCut_;
  unsigned int min_Ntrks_;

  bool resetHistos_;
  // ----------member data ---------------------------
  
  //   std::vector<BSTrkParameters> fBSvector;
  reco::BeamSpot refBS;
  reco::BeamSpot preBS;
  
  // MonitorElements:
  MonitorElement * h_nTrk_lumi;
  MonitorElement * h_d0_phi0;
  MonitorElement * h_x0_lumi;
  MonitorElement * h_y0_lumi;
  MonitorElement * h_z0_lumi;
  MonitorElement * h_sigmaZ0_lumi;
  MonitorElement * h_trk_z0;
  MonitorElement * h_vx_vy;
  MonitorElement * h_vx_dz;
  MonitorElement * h_vy_dz;
  MonitorElement * h_trkPt;
  MonitorElement * h_trkVz;
  MonitorElement * fitResults;
  MonitorElement * h_x0;
  MonitorElement * h_y0;
  MonitorElement * h_z0;
  MonitorElement * h_sigmaZ0;

  // Summary:
  Float_t reportSummary_;
  Float_t summarySum_;
  Float_t summaryContent_[3];
  MonitorElement * reportSummary;
  MonitorElement * reportSummaryContents[3];
  MonitorElement * reportSummaryMap;
  // variables for beam fit

};

#endif

