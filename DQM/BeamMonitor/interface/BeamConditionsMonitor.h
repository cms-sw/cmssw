#ifndef BeamConditionsMonitor_H
#define BeamConditionsMonitor_H

/** \class BeamConditionsMonitor
 * *
 *  $Date: 2009/08/05 14:45:09 $
 *  $Revision: 1.4 $
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
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

//
// class declaration
//

class BeamConditionsMonitor : public edm::EDAnalyzer {
 public:
  BeamConditionsMonitor( const edm::ParameterSet& );
  ~BeamConditionsMonitor();

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
  bool debug_;
  
  DQMStore* dbe_;

  int countEvt_;      //counter
  int countLumi_;      //counter
  
  // ----------member data ---------------------------
  const reco::BeamSpot * theBS;

  // MonitorElements
  MonitorElement * h_x0_lumi;
  MonitorElement * h_y0_lumi;
  
};

#endif
