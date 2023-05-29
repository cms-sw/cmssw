#ifndef DQMOFFLINE_TRIGGER_EGHLTOFFLINESOURCE
#define DQMOFFLINE_TRIGGER_EGHLTOFFLINESOURCE

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

//#include "DataFormats/HLTReco/interface/TriggerEvent.h"
//#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"

#include "DQMOffline/Trigger/interface/EgHLTBinData.h"
#include "DQMOffline/Trigger/interface/EgHLTCutMasks.h"
#include "DQMOffline/Trigger/interface/EgHLTMonElemContainer.h"
#include "DQMOffline/Trigger/interface/EgHLTMonElemFuncs.h"
#include "DQMOffline/Trigger/interface/EgHLTOffHelper.h"
#include "DQMOffline/Trigger/interface/EgHLTOffEvt.h"
#include "DQMOffline/Trigger/interface/EgHLTTrigCodes.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

class HLTConfigProvider;

namespace egHLT {
  class EleHLTFilterMon;
  class PhoHLTFilterMon;
}  // namespace egHLT

// namespace trigger{
//   class TriggerObject;

// }

class EgHLTOfflineSource : public DQMEDAnalyzer {
private:
  MonitorElement* dqmErrsMonElem_;            //monitors DQM errors (ie failing to get trigger info, etc)
  MonitorElement* nrEventsProcessedMonElem_;  //number of events processed mon elem
  int nrEventsProcessed_;                     //number of events processed

  std::vector<egHLT::EleHLTFilterMon*>
      eleFilterMonHists_;  //monitoring histograms for different trigger paths, we own them
  std::vector<egHLT::PhoHLTFilterMon*>
      phoFilterMonHists_;  //monitoring histograms for different trigger paths, we own them
  std::vector<egHLT::MonElemContainer<egHLT::OffEle>*> eleMonElems_;  //mon elements for monitoring electrons, we own them
  std::vector<egHLT::MonElemContainer<egHLT::OffPho>*> phoMonElems_;  //mon elements for monitoring photons, we own them

  std::string dirName_;
  std::string subdirName_;
  bool dohep_;
  egHLT::OffEvt offEvt_;
  egHLT::OffHelper
      offEvtHelper_;  // this is where up wrap up nasty code which will be replaced by offical tools at some point
  std::unique_ptr<egHLT::TrigCodes> trigCodes;  // the only place instantiate them

  //note ele,pho does not refer to whether the trigger is electron or photon, it refers to what type
  //of object passing the trigger will be monitored, eg ele = offline gsf electrons
  std::vector<std::string> eleHLTFilterNames_;  //names of the filter names to use for offline ele
  std::vector<std::string> phoHLTFilterNames_;  //names of the filter names to use for offline pho
  std::vector<std::string>
      eleHLTFilterNames2Leg_;  //names of the first leg of 2Leg filter names to use for offline ele in form filter1::filter2
  std::vector<std::string> eleTightLooseTrigNames_;    // list of 'tightTrig:looseTrig' pairs for ele
  std::vector<std::string> phoTightLooseTrigNames_;    // list of 'tightTrig:looseTrig' pairs for pho
  std::vector<std::string> diEleTightLooseTrigNames_;  // list of 'tightTrig:looseTrig' pairs for di-ele triggers
  std::vector<std::string> diPhoTightLooseTrigNames_;  // list of 'tightTrig:looseTrig' pairs for di-pho triggers

  egHLT::BinData binData_;
  egHLT::CutMasks cutMasks_;

  bool filterInactiveTriggers_;
  std::string hltTag_;

public:
  explicit EgHLTOfflineSource(const edm::ParameterSet&);

  //disabling copying/assignment (copying this class would be bad, mkay)
  EgHLTOfflineSource(const EgHLTOfflineSource& rhs) = delete;
  EgHLTOfflineSource& operator=(const EgHLTOfflineSource& rhs) = delete;

  ~EgHLTOfflineSource() override;

  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  void addEleTrigPath(egHLT::MonElemFuncs& monElemFuncs, const std::string& name);
  void addPhoTrigPath(egHLT::MonElemFuncs& monElemFuncs, const std::string& name);
  void getHLTFilterNamesUsed(std::vector<std::string>& filterNames) const;
  void filterTriggers(const HLTConfigProvider& hltConfig);
};

#endif
