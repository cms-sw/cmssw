#ifndef BeamMonitorBx_H
#define BeamMonitorBx_H

/** \class BeamMonitorBx
 * *
 *  $Date: 2010/07/29 22:15:24 $
 *  $Revision: 1.8 $
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

  typedef int BxNum;
  typedef std::map<BxNum,reco::BeamSpot> BeamSpotMapBx;

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

  void FitAndFill(const edm::LuminosityBlock& lumiSeg, int&, int&, int&);
  void BookTables(int, std::map<std::string,std::string>&,std::string);
  void BookTrendHistos(bool, int, std::map<std::string,std::string>&, 
		       std::string, TString, TString);
  void FillTables(int, int, std::map<std::string,std::string>&,
		  reco::BeamSpot&, std::string);
  void FillTrendHistos(int, int, std::map<std::string,std::string>&,
		       reco::BeamSpot&, TString);
  void weight(BeamSpotMapBx&, const BeamSpotMapBx&);
  void weight(double& mean,double& meanError,const double& val,const double& valError);
  const char * formatFitTime( const std::time_t&);

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
  int firstlumi_; // first LS with good fit
  int countGoodFit_;
  std::time_t refBStime[2];

  bool resetHistos_;
  bool processed_;
  // ----------member data ---------------------------
  BeamSpotMapBx fbspotMap;//for weighted beam spots of each bunch
  std::map<std::string, std::string> varMap;
  std::map<std::string, std::string> varMap1;
  // MonitorElements:
  std::map<TString, MonitorElement*> hs; // Tables
  std::map<TString, MonitorElement*> hst; // Trending Histos

  //Test
  //  MonitorElement * h_x0;

  //
  std::time_t tmpTime;
  std::time_t refTime;
  std::time_t startTime;
  edm::TimeValue_t ftimestamp;

};

#endif

