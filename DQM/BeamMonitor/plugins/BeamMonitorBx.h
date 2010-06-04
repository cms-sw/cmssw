#ifndef BeamMonitorBx_H
#define BeamMonitorBx_H

/** \class BeamMonitorBx
 * *
 *  $Date: 2010/06/02 05:02:40 $
 *  $Revision: 1.2 $
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
#include "RecoVertex/BeamSpotProducer/interface/BeamFitter.h"
#include <fstream>


//
// class declaration
//

class BeamMonitorBx : public edm::EDAnalyzer {
 public:
  BeamMonitorBx( const edm::ParameterSet& );
  ~BeamMonitorBx();

 protected:
   
  // BeginJob
  void beginJob();

  // BeginRun
  void beginRun(const edm::Run& r, const edm::EventSetup& c);
  
  void analyze(const edm::Event& e, const edm::EventSetup& c) ;
  
  void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
			    const edm::EventSetup& context) ;
  
  void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
			  const edm::EventSetup& c);
  // EndRun
  void endRun(const edm::Run& r, const edm::EventSetup& c);
  // Endjob
  void endJob(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& c);
  
 private:

  void FitAndFill(const edm::LuminosityBlock& lumiSeg,int&,int&,int&);
  void BookHistos(int, std::string);

  edm::ParameterSet parameters_;
  std::string monitorName_;
  edm::InputTag bsSrc_; // beam spot

  int fitNLumi_;
  int resetFitNLumi_;
  bool debug_;
  
  DQMStore* dbe_;
  BeamFitter * theBeamFitter;
  
  unsigned int countBx_;
  int countEvt_;       //counter
  int countLumi_;      //counter
  int beginLumiOfBSFit_;
  int endLumiOfBSFit_;
  int lastlumi_; // previous LS processed
  int nextlumi_; // next LS of Fit
  std::time_t refBStime[2];

  bool resetHistos_;
  bool processed_;
  // ----------member data ---------------------------
  std::map<int, reco::BeamSpot> fbspotMap;
  std::map<std::string, std::string> varMap;
  // MonitorElements:
  std::map<TString, MonitorElement*> hs;

  //
  std::time_t tmpTime;
  std::time_t refTime;
  edm::TimeValue_t ftimestamp;

};

#endif

