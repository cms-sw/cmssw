#ifndef DQMOFFLINE_TRIGGER_EGHLTOFFLINECLIENT
#define DQMOFFLINE_TRIGGER_EGHLTOFFLINECLIENT

// -*- C++ -*-
//
// Package:    EgammaHLTOfflineClient
// Class:      EgammaHLTOffline
// 
/*
 Description: This is a DQM client meant to plot high-level HLT trigger 
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

#include "DQMOffline/Trigger/interface/MonElemManager.h"
#include "DQMOffline/Trigger/interface/EgHLTOffHelper.h"

class DQMStore;

class EleHLTPathMon;

namespace trigger{
  class TriggerObject;

}

class EgHLTOfflineClient : public edm::EDAnalyzer {
 
 private:
  DQMStore* dbe_; //dbe seems to be the standard name for this, I dont know why. We of course dont own it
  std::string dirName_;
  std::vector<std::string> eleHLTPathNames_;//names of the HLT paths to use
  std::vector<std::string> eleHLTFilterNames_;//names of the filter names to use, appended to the pathNames 
  std::vector<std::string> eleHLTTightLooseFilters_;//names of the filter names to use, appended to the pathNames
  //disabling copying/assignment
  EgHLTOfflineClient(const EgHLTOfflineClient& rhs){}
  EgHLTOfflineClient& operator=(const EgHLTOfflineClient& rhs){return *this;}

 public:
  explicit EgHLTOfflineClient(const edm::ParameterSet& );
  virtual ~EgHLTOfflineClient();
  
  
  virtual void beginJob(const edm::EventSetup&);
  virtual void analyze(const edm::Event&, const edm::EventSetup&); //dummy
  virtual void endJob();
  virtual void beginRun(const edm::Run& run, const edm::EventSetup& c);
  virtual void endRun(const edm::Run& run, const edm::EventSetup& c);
  
  
  virtual void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg,const edm::EventSetup& context){}
  // DQM Client Diagnostic
  virtual void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg,const edm::EventSetup& c);

  //at somepoint these all may migrate to a helper class
  void createN1EffHists(const std::string& baseName,const std::string& region="");
  void createLooseTightTrigEff(const std::string& filterName,const std::string& region="");
  void createTrigTagProbeEffHists(const std::string& filterName,const std::string& region="");
  MonitorElement* makeEffMonElemFromPassAndAll(const std::string& name,const MonitorElement* pass,const MonitorElement* all);
  MonitorElement* makeEffMonElemFromPassAndFail(const std::string& name,const MonitorElement* pass,const MonitorElement* fail);
  
};
 


#endif
