#ifndef DQM_L1TMonitor_L1TStage2mGMT_h
#define DQM_L1TMonitor_L1TStage2mGMT_h

/*
 * \file L1TStage2mGMT.h
 * \Author Esmaeel Eskandari Tadavani
*/

// system requirements
#include <iosfwd>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>

// general requirements
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/OrphanHandle.h"

// stage2 requirements
#include "DataFormats/L1Trigger/interface/BXVector.h"
#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"

// dqm requirements
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

// class decleration

class L1TStage2mGMT: public DQMEDAnalyzer {

public:

// class constructor
L1TStage2mGMT(const edm::ParameterSet & ps);

// class destructor
virtual ~L1TStage2mGMT();

// member functions
protected:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) override;
  virtual void dqmBeginRun(const edm::Run&, const edm::EventSetup&) override;
  virtual void bookHistograms(DQMStore::IBooker&, const edm::Run&, const edm::EventSetup&) override ;

// data members
private:  

  std::string monitorDir;
  edm::InputTag stage2mgmtSource ; 
  bool verbose ;
  edm::EDGetToken stage2mgmtToken ; 

  MonitorElement* eta_mgmt;
  MonitorElement* phi_mgmt;
  MonitorElement* pt_mgmt;
  MonitorElement* charge_mgmt;
  MonitorElement* chargeVal_mgmt;
  MonitorElement* qual_mgmt;
  MonitorElement* iso_mgmt;

  MonitorElement* bx_mgmt;

  MonitorElement* etaVSbx_mgmt;
  MonitorElement* phiVSbx_mgmt;
  MonitorElement* ptVSbx_mgmt;
  MonitorElement* chargeVSbx_mgmt;
  MonitorElement* chargeValVSbx_mgmt;
  MonitorElement* qualVSbx_mgmt;
  MonitorElement* isoVSbx_mgmt;

  MonitorElement* etaVSphi_mgmt;
  MonitorElement* phiVSpt_mgmt;
  MonitorElement* etaVSpt_mgmt;




};

#endif
