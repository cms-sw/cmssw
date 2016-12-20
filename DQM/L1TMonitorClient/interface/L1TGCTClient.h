#ifndef DQM_L1TMONITORCLIENT_L1TGCTCLIENT_H
#define DQM_L1TMONITORCLIENT_L1TGCTCLIENT_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"

#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

class L1TGCTClient: public DQMEDHarvester {

 public:

  /// Constructor
  L1TGCTClient(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~L1TGCTClient();
 
 protected:
  virtual void dqmEndJob(DQMStore::IBooker &ibooker,DQMStore::IGetter &igetter) override;
  virtual void dqmEndLuminosityBlock(DQMStore::IBooker &ibooker,DQMStore::IGetter &igetter,const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& c);

 private:

  void book(DQMStore::IBooker &ibooker);
  void processHistograms(DQMStore::IGetter &igetter);

  void makeXProjection(TH2F* input, MonitorElement* output);
  void makeYProjection(TH2F* input, MonitorElement* output);

  std::string monitorDir_; 
  int counterLS_;      ///counter 
  int counterEvt_;     ///counter 
  int prescaleLS_;     ///units of lumi sections 
  int prescaleEvt_;    ///prescale on number of events

  bool m_runInEventLoop;
  bool m_runInEndLumi;
  bool m_runInEndRun;
  bool m_runInEndJob;
  bool m_stage1_layer2_;

  std::string InputDir;

  MonitorElement *l1GctIsoEmOccEta_;
  MonitorElement *l1GctIsoEmOccPhi_;
  MonitorElement *l1GctNonIsoEmOccEta_;
  MonitorElement *l1GctNonIsoEmOccPhi_;
  MonitorElement *l1GctAllJetsOccEta_;
  MonitorElement *l1GctAllJetsOccPhi_;
  MonitorElement *l1GctCenJetsOccEta_;
  MonitorElement *l1GctCenJetsOccPhi_;
  MonitorElement *l1GctForJetsOccEta_;
  MonitorElement *l1GctForJetsOccPhi_;
  MonitorElement *l1GctTauJetsOccEta_;
  MonitorElement *l1GctTauJetsOccPhi_;
  MonitorElement *l1GctIsoTauJetsOccEta_;
  MonitorElement *l1GctIsoTauJetsOccPhi_;

};

#endif
