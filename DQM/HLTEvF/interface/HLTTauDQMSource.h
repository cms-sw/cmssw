#ifndef HLTTauDQMSource_H
#define HLTTauDQMSource_H

/*DQM For Tau HLT
Author : Michail Bachtis
University of Wisconsin-Madison
bachtis@hep.wisc.edu

Derived by HLTMuonDQMSource.h
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
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "Math/GenVector/VectorUtil.h"

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


//L2 Tau trigger Includes
#include "DataFormats/TauReco/interface/L2TauInfoAssociation.h"

//L25Tau Trigger Includes
#include "DataFormats/BTauReco/interface/IsolatedTauTagInfo.h"
#include "DataFormats/TrackReco/interface/Track.h"

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
  void beginJob(const edm::EventSetup& c);

  /// BeginRun
  void beginRun(const edm::Run& r, const edm::EventSetup& c);

  /// Fake Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c) ;

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

  /* GENERAL DQM PATH */
  
  //Set the Monitor Parameters
  std::string mainFolder_; //main DQM Folder
  std::string monitorName_;///Monitor name
  std::string outputFile_;///OutputFile
  int counterEvt_;      ///counter
  int prescaleEvt_;     ///every n events 
  bool disable_;        ///disable
  bool verbose_;          
  unsigned  nTriggeredTaus_;
  double EtMin_;
  double EtMax_;
  int NEtBins_;
  int NEtaBins_;

  //get The Jet Collections per filter level
  edm::InputTag l1Filter_;
  edm::InputTag l2Filter_;
  edm::InputTag l25Filter_;
  edm::InputTag l3Filter_;

  //Correlations with other Triggers
  std::vector<edm::InputTag> refFilters_;
  std::vector<int> refIDs_;
  std::vector<double> PtCut_;
  std::vector<std::string> refFilterDesc_;
  double corrDeltaR_;

  //L2 Monitoring Parameters
  bool doL2Monitoring_;
  edm::InputTag l2AssocMap_;


  //L25 Monitoring Parameters
  bool doL25Monitoring_;
  edm::InputTag l25IsolInfo_;
  double l25LeadTrackDeltaR_;
  double l25LeadTrackPt_;

  //L3 Monitoring Parameters
  bool doL3Monitoring_;
  edm::InputTag l3IsolInfo_;
  double l3LeadTrackDeltaR_;
  double l3LeadTrackPt_;



  //Number of Tau Events passed the triggers
  int NEventsPassedL1;
  int NEventsPassedL2;
  int NEventsPassedL25;
  int NEventsPassedL3;

  //Number of Tau Events passed the triggers matched to reference objects
  std::vector<int> NEventsPassedRefL1;
  std::vector<int> NEventsPassedRefL2;
  std::vector<int> NEventsPassedRefL25;
  std::vector<int> NEventsPassedRefL3;

  //Number of reference objects
  std::vector<int> NRefEvents;

  //For Efficiencies we need to calculate the et and eta for the reference objects
  //  std::vector<TH1F*> EtRef_;
  // std::vector<TH1F*> EtaRef_;



  //MonitorElements(Trigger Bits and Efficiency with ref to L1)
  MonitorElement *triggerBitInfoSum_;
  MonitorElement *triggerBitInfo_;
  MonitorElement *triggerEfficiencyL1_;
  

  //Matching to reference triggers and trigger efficiencies
  std::vector<MonitorElement*> triggerBitInfoRef_;
  std::vector<MonitorElement*> triggerEfficiencyRef_;


  //MonitorElements For L2 -Inclusive
  MonitorElement* L2JetEt_;
  MonitorElement* L2JetEta_;
  MonitorElement* L2EcalIsolEt_;
  MonitorElement* L2TowerIsolEt_;
  MonitorElement* L2SeedTowerEt_;
  MonitorElement* L2NClusters_;
  MonitorElement* L2ClusterEtaRMS_;
  MonitorElement* L2ClusterPhiRMS_;
  MonitorElement* L2ClusterDeltaRRMS_;

  //MonitorElements for L2 - With Matching
  std::vector<MonitorElement*> L2JetEtRef_;
  std::vector<MonitorElement*> L2JetEtaRef_;
  std::vector<MonitorElement*> L2EcalIsolEtRef_;
  std::vector<MonitorElement*> L2TowerIsolEtRef_;
  std::vector<MonitorElement*> L2SeedTowerEtRef_;
  std::vector<MonitorElement*> L2NClustersRef_;
  std::vector<MonitorElement*> L2ClusterEtaRMSRef_;
  std::vector<MonitorElement*> L2ClusterPhiRMSRef_;
  std::vector<MonitorElement*> L2ClusterDeltaRRMSRef_;


  //MonitorElements for L25 Inclusive
  MonitorElement* L25JetEt_;
  MonitorElement* L25JetEta_;
  MonitorElement* L25NPixelTracks_;
  MonitorElement* L25NQPixelTracks_;
  MonitorElement* L25HasLeadingTrack_;
  MonitorElement* L25LeadTrackPt_;
  MonitorElement* L25SumTrackPt_;

  //MonitorElements for L25 (with matching)
  std::vector<MonitorElement*> L25JetEtRef_;
  std::vector<MonitorElement*> L25JetEtaRef_;
  std::vector<MonitorElement*> L25NPixelTracksRef_;
  std::vector<MonitorElement*> L25NQPixelTracksRef_;
  std::vector<MonitorElement*> L25HasLeadingTrackRef_;
  std::vector<MonitorElement*> L25LeadTrackPtRef_;
  std::vector<MonitorElement*> L25SumTrackPtRef_;



  //MonitorElements for L3 Inclusive
  MonitorElement* L3JetEt_;
  MonitorElement* L3JetEta_;
  MonitorElement* L3NPixelTracks_;
  MonitorElement* L3NQPixelTracks_;
  MonitorElement* L3HasLeadingTrack_;
  MonitorElement* L3LeadTrackPt_;
  MonitorElement* L3SumTrackPt_;

  //MonitorElements for L3 (with matching)
  std::vector<MonitorElement*> L3JetEtRef_;
  std::vector<MonitorElement*> L3JetEtaRef_;
  std::vector<MonitorElement*> L3NPixelTracksRef_;
  std::vector<MonitorElement*> L3NQPixelTracksRef_;
  std::vector<MonitorElement*> L3HasLeadingTrackRef_;
  std::vector<MonitorElement*> L3LeadTrackPtRef_;
  std::vector<MonitorElement*> L3SumTrackPtRef_;





  
  //Efficiencies with ref to L1 - Inclusive
  //  MonitorElement* L2EtEffL1;
  //  MonitorElement* L2EtaEffL1;

  //Efficiencies with ref to Reference Triggers
  //std::vector<MonitorElement*> L2EtEffRef;
  //std::vector<MonitorElement*> L2EtaEffRef;


  
  
  

  //HELPER FUNCTIONS
  void doSummary(const edm::Event& e, const edm::EventSetup& c);
  void doL2(const edm::Event& e, const edm::EventSetup& c);
  void doL25(const edm::Event& e, const edm::EventSetup& c);
  void doL3(const edm::Event& e, const edm::EventSetup& c);


  bool match(const LV&,const LVColl& /*trigger::VRelectron&*/,double,double);
  std::vector<double> calcEfficiency(int,int);
  LVColl importObjectColl(edm::InputTag&,int,const edm::Event&);

  void formatHistogram(MonitorElement*,int);


};

#endif

