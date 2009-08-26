#ifndef BeamMonitor_H
#define BeamMonitor_H

/** \class BeamMonitor
 * *
 *  $Date: 2009/08/25 21:46:35 $
 *  $Revision: 1.1 $
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
  void beginJob(const edm::EventSetup& c);

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
  void endJob();
  
 private:
  
  edm::ParameterSet parameters_;
  std::string monitorName_;
  std::string bsSrc_; // beam spot
  int fitNLumi_;
  int resetFitNLumi_;
  bool debug_;
  
  DQMStore* dbe_;
  BeamFitter * theBeamFitter;
  
  int countEvt_;       //counter
  int countLumi_;      //counter
  int nthBSTrk_;       //

  bool resetHistos_;
  // ----------member data ---------------------------
  
  //   std::vector<BSTrkParameters> fBSvector;
  
  // MonitorElements:
  MonitorElement * h_nTrk_lumi;
  MonitorElement * h_d0_phi0;
  MonitorElement * h_x0_lumi;
  MonitorElement * h_y0_lumi;
  MonitorElement * h_vx_vy;
  
  // variables for beam fit

};

#endif

