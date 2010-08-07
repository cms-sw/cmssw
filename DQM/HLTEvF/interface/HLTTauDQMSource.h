/*DQM For Tau HLT
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

#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerRefsCollections.h"
#include "DataFormats/Math/interface/LorentzVector.h"

//MET Includes
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"

//Electron includes
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"

//Muon Includes
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"

//Photon Includes
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"

//Track Include
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidateFwd.h"
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h"

//Plotters
#include "DQM/HLTEvF/interface/HLTTauDQML1Plotter.h"
#include "DQM/HLTEvF/interface/HLTTauDQMCaloPlotter.h"
#include "DQM/HLTEvF/interface/HLTTauDQMTrkPlotter.h"
#include "DQM/HLTEvF/interface/HLTTauDQMPathPlotter.h"
#include "DQM/HLTEvF/interface/HLTTauDQMSummaryPlotter.h"
#include "DQM/HLTEvF/interface/HLTTauDQMLitePathPlotter.h"

//
// class declaration
//

typedef math::XYZTLorentzVectorD LV;
typedef std::vector<LV> LVColl;

class HLTTauDQMSource : public edm::EDAnalyzer {
public:
  HLTTauDQMSource( const edm::ParameterSet& );
  ~HLTTauDQMSource();

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
  std::vector<std::string> configType_;

  //Reference
  bool doRefAnalysis_;
  int NPtBins_;
  int NEtaBins_;
  int NPhiBins_;
  double EtMax_;
  double L1MatchDr_;
  double HLTMatchDr_;


  std::vector<edm::InputTag> refFilter_;
  std::vector<int> refID_;
  std::vector<double> ptThres_;

  //  edm::InputTag refFilter2_;
  //  int refID2_;
  //  double ptThres2_;


  //DQM Prescaler
  int counterEvt_;      ///counter
  int prescaleEvt_;     ///every n events 

  //get The trigger Event
  edm::InputTag triggerEvent_;
  //Helper function to get Trigger event primitives
  //  LVColl getFilterCollection(size_t,int,const trigger::TriggerEventWithRefs&,double);
  LVColl getFilterCollection(size_t index,int id,const trigger::TriggerEvent& trigEv,double);


  //Define Dummy vectors of Plotters
  std::vector<HLTTauDQML1Plotter> l1Plotters;
  std::vector<HLTTauDQMCaloPlotter> caloPlotters;
  std::vector<HLTTauDQMTrkPlotter> trackPlotters; 
  std::vector<HLTTauDQMPathPlotter> pathPlotters;
  std::vector<HLTTauDQMLitePathPlotter> litePathPlotters;
  std::vector<HLTTauDQMSummaryPlotter> summaryPlotters;

};

