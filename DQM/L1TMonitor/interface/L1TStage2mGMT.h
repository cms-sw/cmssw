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

  MonitorElement* hw_eta_ugmt;
  MonitorElement* hw_phi_ugmt;
  MonitorElement* hw_pt_ugmt;

  MonitorElement* ph_eta_ugmt;
  MonitorElement* ph_phi_ugmt;
  MonitorElement* ph_pt_ugmt;

  MonitorElement* hw_etaVSphi_ugmt;
  MonitorElement* hw_phiVSpt_ugmt;
  MonitorElement* hw_etaVSpt_ugmt;

  MonitorElement* ph_etaVSphi_ugmt;
  MonitorElement* ph_phiVSpt_ugmt;
  MonitorElement* ph_etaVSpt_ugmt;

  MonitorElement* charge_ugmt;
  MonitorElement* chargeVal_ugmt;
  MonitorElement* qual_ugmt;
  MonitorElement* iso_ugmt;

  MonitorElement* bx_ugmt;

  MonitorElement* hw_etaVSbx_ugmt;
  MonitorElement* hw_phiVSbx_ugmt;
  MonitorElement* hw_ptVSbx_ugmt;

  MonitorElement* ph_etaVSbx_ugmt;
  MonitorElement* ph_phiVSbx_ugmt;
  MonitorElement* ph_ptVSbx_ugmt;

  MonitorElement* chargeVSbx_ugmt;
  MonitorElement* chargeValVSbx_ugmt;
  MonitorElement* qualVSbx_ugmt;
  MonitorElement* isoVSbx_ugmt;

};

#endif
