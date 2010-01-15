#ifndef HLTMonBitSummary_H
#define HLTMonBitSummary_H


// system include files
#include <memory>
#include <unistd.h>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"

#include "FWCore/Framework/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Run.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


class HLTMonBitSummary : public edm::EDAnalyzer {
 public:
  explicit HLTMonBitSummary(const edm::ParameterSet&);
  ~HLTMonBitSummary();
  
 private:
  virtual void beginJob();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  virtual void beginRun(const edm::Run  & r, const edm::EventSetup  &);
  //  virtual void endRun(const edm::Run &, const edm::EventSetup &);

  edm::InputTag inputTag_;
  edm::TriggerNames triggerNames_;
  DQMStore * dbe_;

  std::vector<std::string > HLTPathsByName_;
  std::vector<std::string > HLTPathNamesConfig_;
  std::vector<std::string > HLTPathNamesConfigPreVal_;
  std::vector<std::string > HLTPathNamesKey_;

  std::vector<std::string > HLTPathDenomName_;
  std::vector<std::string > filterTypes_;
  std::vector<unsigned int> HLTPathsByIndex_;
  std::string denominator_;
  std::string denominatorWild_;
  std::vector<std::string > dummyFilters_;
  std::string esPathsKey_;


  std::vector<unsigned int> count_;

  std::vector <std::vector <std::string> > triggerFilters_;
  std::vector <std::vector <uint> > triggerFilterIndices_;

  bool configFlag_;
  bool filterFlag_;

  unsigned int total_;
  unsigned int nValidTriggers_;
  unsigned int nValidConfigTriggers_;
  //static const int NTRIG = 20;

  //std::string out_;
  unsigned int ndenomAccept_;

  std::string directory_;
  std::string histLabel_;
  //std::string label_;

  //MonitorElement * hEffSummary;
  //MonitorElement * hCountSummary;
  //MonitorElement * hSubFilterCount[NTRIG];
  //MonitorElement * hSubFilterEff[NTRIG];
  std::vector<MonitorElement* > hSubFilterCount;
  std::vector<MonitorElement* > hSubFilterEff;

  MonitorElement * h1_;
  MonitorElement * h2_;
  MonitorElement * pf_;
  MonitorElement * ratio_;

};
#endif
