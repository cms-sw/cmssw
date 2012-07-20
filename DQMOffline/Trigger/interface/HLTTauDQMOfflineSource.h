/*Offlone DQM For Tau HLT
Author : Michail Bachtis
University of Wisconsin-Madison
bachtis@hep.wisc.edu
*/

 
#include <memory>
#include <unistd.h>
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Math/interface/LorentzVector.h"
//Plotters
#include "DQM/HLTEvF/interface/HLTTauDQML1Plotter.h"
#include "DQM/HLTEvF/interface/HLTTauDQMCaloPlotter.h"
#include "DQM/HLTEvF/interface/HLTTauDQMTrkPlotter.h"
#include "DQM/HLTEvF/interface/HLTTauDQMPathPlotter.h"
#include "DQM/HLTEvF/interface/HLTTauDQMLitePathPlotter.h"

//
// class declaration
//

typedef math::XYZTLorentzVectorD LV;
typedef std::vector<LV> LVColl;

class HLTTauDQMOfflineSource : public edm::EDAnalyzer {
public:
  HLTTauDQMOfflineSource( const edm::ParameterSet& );
  ~HLTTauDQMOfflineSource();

protected:
   
  /// BeginJob
  void beginJob();

  /// BeginRun
  void beginRun(const edm::Run& r, const edm::EventSetup& c);

  /// Fake Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c) ;

  ///Luminosity Block 
  void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                            const edm::EventSetup& context) ;
  /// DQM Client Diagnostic
  void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                          const edm::EventSetup& c);
  /// EndRun
  void endRun(const edm::Run& r, const edm::EventSetup& c);

  /// Endjob
  void endJob();



private:
  DQMStore* dbe_;  
  std::vector<edm::ParameterSet> config_;

  //Reference
  bool doRefAnalysis_;
  int NPtBins_;
  int NEtaBins_;
  int NPhiBins_;
  double EtMax_;
  double L1MatchDr_;
  double HLTMatchDr_;


  std::vector<edm::InputTag> refObjects_;


  //  edm::InputTag refFilter2_;
  //  int refID2_;
  //  double ptThres2_;


  //DQM Prescaler
  int counterEvt_;      ///counter
  int prescaleEvt_;     ///every n events 

  //get The trigger Event
  edm::InputTag triggerEvent_;

  //Define Dummy vectors of Plotters
  std::vector<HLTTauDQML1Plotter> l1Plotters;
  std::vector<HLTTauDQMCaloPlotter> caloPlotters;
  std::vector<HLTTauDQMTrkPlotter> trackPlotters; 
  std::vector<HLTTauDQMPathPlotter> pathPlotters;
  std::vector<HLTTauDQMLitePathPlotter> litePathPlotters;
};

