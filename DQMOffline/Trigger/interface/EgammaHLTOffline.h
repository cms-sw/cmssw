#ifndef DQMOFFLINE_TRIGGER_EGAMMAHLTOFFLINE
#define DQMOFFLINE_TRIGGER_EGAMMAHLTOFFLINE

// -*- C++ -*-
//
// Package:    EgammaHLTOffline
// Class:      EgammaHLTOffline
// 
/*
 Description: This is a DQM source meant to plot high-level HLT trigger 
 quantities as stored in the HLT results object TriggerResults for the Egamma triggers

 Notes:
  Currently I would like to plot simple histograms of three seperate types of variables
  1) global event quantities: eg nr of electrons
  2) di-object quanities: transverse mass, di-electron mass
  3) single object kinematic and id variables: eg et,eta,isolation

*/
//
// Original Author:  Sam Harper
//         Created:  June 2008
// 
//
//

#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"

#include "DQMOffline/Trigger/interface/MonElemContainer.h"
#include "DQMOffline/Trigger/interface/MonElemManager.h"
#include "DQMOffline/Trigger/interface/EgHLTOffHelper.h"
#include "DQMOffline/Trigger/interface/TrigCodes.h"

class DQMStore;

class EleHLTPathMon;

namespace trigger{
  class TriggerObject;

}

class EgammaHLTOffline : public edm::EDAnalyzer {
 
 private:
  DQMStore* dbe_; //dbe seems to be the standard name for this, I dont know why. We of course dont own it
  
  std::vector<EleHLTPathMon*> elePathMonHists_; //monitoring histograms for different trigger paths, we own them
  edm::InputTag triggerSummaryLabel_;
  std::string dirName_;
  
  EgHLTOffHelper egHelper_;// this is where up wrap up nasty code which will be replaced by offical tools at some point
  std::vector<std::string> namesFiltersUsed_; //the names of all the filters used (so we dont have to do calculations for every filter in the event

  std::vector<std::string> eleHLTPathNames_;//names of the HLT paths to use
  std::vector<std::string> eleHLTFilterNames_;//names of the filter names to use, appended to the pathNames

  std::vector<MonElemContainer<EgHLTOffEle>*> eleMonElems_;

  //disabling copying/assignment
  EgammaHLTOffline(const EgammaHLTOffline& rhs){}
  EgammaHLTOffline& operator=(const EgammaHLTOffline& rhs){return *this;}

 public:
  explicit EgammaHLTOffline(const edm::ParameterSet& );
  virtual ~EgammaHLTOffline();
  
  
  virtual void beginJob(const edm::EventSetup&);
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob();
  virtual void beginRun(const edm::Run& run, const edm::EventSetup& c);
  virtual void endRun(const edm::Run& run, const edm::EventSetup& c);

  void addTrigPath(const std::string& name);
  void obtainFiltersElePasses(const std::vector<EgHLTOffEle>& eles,const std::vector<std::string>& filters,edm::Handle<trigger::TriggerEvent> trigEvt,std::vector<std::vector<int> >& filtersElePasses);
  void filterNamesUsed(std::vector<std::string>& filterNames);
  TrigCodes::TrigBitSet setFiltersElePasses(std::vector<EgHLTOffEle>& eles,const std::vector<std::string>& filters,edm::Handle<trigger::TriggerEvent> trigEvt);

};
 


#endif
