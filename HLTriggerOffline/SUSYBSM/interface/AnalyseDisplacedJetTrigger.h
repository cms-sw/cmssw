#ifndef AnalyseDisplacedJetTrigger_H
#define AnalyseDisplacedJetTrigger_H
// -*- C++ -*-
//
// Package:    AnalyseDisplacedJetTrigger
// Class:      AnalyseDisplacedJetTrigger
// 
/**\class AnalyseDisplacedJetTrigger AnalyseDisplacedJetTrigger.cc
 Description: EDAnalyzer to analyze Displaced Jet Exotica Trigger.

     Produces one histogram showing the trigger efficiency of displacedJet triggers and of normal jet/HT triggers
     so they can be compared.

     Also produces histograms of offline jet properties for those jets which are selected in offline 
     exotica searches for displaced jets. (i.e. High et jets with few prompt tracks).

     Makes these same histograms also for these jets if they are matched to trigger jets, so
     the efficiency can be understood.

     When making these histograms of jet properties, no cut is applied on the quantity being plotted.
     This allows one to understand if the trigger is firing on events outside the region interesting
     for physics.

     There is also one pair of histograms showing the true production radius of the jets, for all
     jets and those matched to a trigger jet. This allows one to understand efficiency vs. the  
     radius at which a long-lived exotic decayed.

     Will give meaninful efficiency and background info when run on either QCD MC or real data, 
     and more detailed efficiency info when run on displaced jet signal MC.

     Warning: Since the intermediate displaced jet trigger levels require two jets with
     prompt tracks, part of any inefficiency you see will be caused by the requirement
     on the second jet (particularly in data or QCD MC events).
*/
//
// Original Author:  Ian Tomalin
// Date: Feb. 2011
//

// standard EDAnalyser include files
#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// For histograms ...

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

// CMSSW EDProducts
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/TriggerEvent.h"

#include "HLTriggerOffline/SUSYBSM/interface/HistosDisplacedJetTrigger.h"

#include <string>
#include <map>
using namespace std;
using namespace edm;
using namespace pat;

class AnalyseDisplacedJetTrigger : public edm::EDAnalyzer {

public:

  explicit AnalyseDisplacedJetTrigger(const edm::ParameterSet& iConfig);

  ~AnalyseDisplacedJetTrigger() {}
  
private:


  void beginJob() {}

  void beginRun(const edm::Run& run, const edm::EventSetup& c);

  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);

  void endRun(const edm::Run& run, const edm::EventSetup& c) {}

  void endJob();

private:

  // Book histograms to study performance of given trigger.
  void bookHistos(string trigName);

  // Analyse triggers and return trigger objects for displaced jet triggers.
  map<string, TriggerObjectRefVector> getTriggerInfo();

  // Find closest trigger jet to reco jet, if any.
  TriggerObjectRef matchJets(pat::JetRef recoJet, const TriggerObjectRefVector& trigJets);

  // Determine trigger type
  bool isTrigDisplaced(string name) {return (name.find("DisplacedJet") != string::npos);}
  bool isTrigJet(string name) {return (name.substr(0,7) == "HLT_Jet");}
  bool isTrigHT(string name) {return (name.substr(0,6) == "HLT_HT");}

  // Debug printout about primary vertices.
  void debugPrintPV(const edm::Event& iEvent);

private:

  Handle<JetCollection> patJets_;
  Handle<TriggerEvent> patTriggerEvent_;
  Handle<reco::VertexCollection> primaryVertex_;

  // This stores all interesting jet/HT trigger types found so far.
  map<string, int> trigNameList_;

  DQMStore * dbe_;

  // Trigger efficiency for all trigger bits
  MonitorElement* trigEffi_;

  // Histograms for studying performance each displaced jet trigger.
  map<string, HistosDisplacedJetTrigger> histos_;
};
#endif
