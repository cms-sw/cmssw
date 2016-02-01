#ifndef DQM_L1TMonitor_L1TStage2BMTF_h
#define DQM_L1TMonitor_L1TStage2BMTF_h

/*
 * \file L1TStage2BMTF.h
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

class  L1TStage2BMTF: public DQMEDAnalyzer {

public:

// class constructor
L1TStage2BMTF(const edm::ParameterSet & ps);
// class destructor
virtual ~L1TStage2BMTF();

// member functions
protected:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) override;
  virtual void dqmBeginRun(const edm::Run&, const edm::EventSetup&) override;
  virtual void bookHistograms(DQMStore::IBooker&, const edm::Run&, const edm::EventSetup&) override ;

// data members
private:  
  //  enum ensubs {BMTF, OMTF, EMTF, mGMT};

  std::string monitorDir;
  edm::InputTag stage2bmtfSource ; 
  bool verbose ;
  edm::EDGetToken stage2bmtfToken ; 


  MonitorElement* eta_bmtf;
  MonitorElement* phi_bmtf;
  MonitorElement* pt_bmtf;
  MonitorElement* bx_bmtf;
  MonitorElement* etaVSphi_bmtf;
  MonitorElement* phiVSpt_bmtf;
  MonitorElement* etaVSpt_bmtf;
  MonitorElement* etaVSbx_bmtf;
  MonitorElement* phiVSbx_bmtf;
  MonitorElement* ptVSbx_bmtf;


};

#endif
