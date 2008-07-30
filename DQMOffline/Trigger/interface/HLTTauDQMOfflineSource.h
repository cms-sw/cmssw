#ifndef HLTTauDQMOfflineSource_H
#define HLTTauDQMOfflineSource_H

/*Offline DQM For Tau HLT
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
#include "DataFormats/HLTReco/interface/TriggerEvent.h"



typedef math::XYZTLorentzVectorD LV;
typedef std::vector<LV> LVColl;


class HLTTauDQMOfflineSource : public edm::EDAnalyzer {
public:
  HLTTauDQMOfflineSource( const edm::ParameterSet& );
  ~HLTTauDQMOfflineSource();

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
  bool verbose_;        //verbose  

  unsigned  nTriggeredTaus_; //Number of Taus required by the Trigger
  unsigned  nTriggeredLeptons_; //Number of Taus required by the Trigger
  int leptonPdgID_;
  int tauPdgID_;

  //Trigger Event Object
  edm::InputTag triggerEvent_;
  

  //Tau Paths
  edm::InputTag mainPath_;
  edm::InputTag l1BackupPath_;
  edm::InputTag l2BackupPath_;
  edm::InputTag l25BackupPath_;
  edm::InputTag l3BackupPath_;

  //Correlations with other Objects
  edm::InputTag refTauObjects_; //Reference object collections for Taus
  edm::InputTag refLeptonObjects_; //Reference object collections for Leptons
  double corrDeltaR_; // Delta R to match to offline



  //Et Histogram Limits
  double EtMin_;
  double EtMax_;
  int NEtBins_;
  int NEtaBins_;


  //Number of reference objects
  int NRefEvents; //Reference Reco events

   
  //Path Monitoring
  MonitorElement* EventsPassed_;
  MonitorElement* EventsPassedMatched_;
  MonitorElement* EventsPassedNotMatched_;
  MonitorElement* EventsRef_;

  //Turn On Curves



  bool match(const LV&,const LVColl& ,double);
  LVColl importFilterColl(edm::InputTag& filter,int pdgID,const edm::Event& iEvent);
 

};

#endif

