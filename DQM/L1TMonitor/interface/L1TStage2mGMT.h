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

// dqm requirements
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"


// class decleration

class  L1TStage2mGMT: public DQMEDAnalyzer {

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
  //  enum ensubs {BMTF, OMTF, EMTF, mGMT};

  std::string monitorDir;
  edm::InputTag stage2mgmtSource ; 
  bool verbose ;
  edm::EDGetToken stage2mgmtToken ; 



  MonitorElement* eta_mgmt;
  MonitorElement* phi_mgmt;
  MonitorElement* pt_mgmt;
  MonitorElement* bx_mgmt;
  MonitorElement* etaVSphi_mgmt;
  MonitorElement* phiVSpt_mgmt;
  MonitorElement* etaVSpt_mgmt;
  MonitorElement* etaVSbx_mgmt;
  MonitorElement* phiVSbx_mgmt;
  MonitorElement* ptVSbx_mgmt;

  // eta & phi 
  //  MonitorElement* subs_eta[4];
  //  MonitorElement* subs_phi[4];
  //  MonitorElement* subs_etaphi[4];
  //  MonitorElement* subs_etaqty[4];

  //  MonitorElement* eta_bmtf_omtf_emtf;
  //  MonitorElement* eta_bmtf;
  //  MonitorElement* eta_omtf;
  //  MonitorElement* eta_emtf;
  //  MonitorElement* phi_bmtf_omtf_emtf;
  //  MonitorElement* phi_bmtf;
  //  MonitorElement* phi_omtf;
  //  MonitorElement* phi_emtf;
  //  MonitorElement* etaphi_bmtf_omtf_emtf;
  /* MonitorElement* etaphi_bmtf; */
  /* MonitorElement* etaphi_omtf; */
  /* MonitorElement* etaphi_emtf; */
  /* MonitorElement* phi_bmtf_omtf; */
  /* MonitorElement* phi_bmtf_emtf; */
  /* MonitorElement* phi_omtf_emtf; */
  /* MonitorElement* eta_bmtf_omtf; */
  /* MonitorElement* eta_bmtf_emtf; */
  /* MonitorElement* eta_omtf_emtf; */

  //  MonitorElement* pt_bmtf;
  //  MonitorElement* pt_omtf;
  //  MonitorElement* pt_emtf;
  // bunch-crossing     
  //  MonitorElement* subs_nbx[4];
  //  MonitorElement* subs_dbx[3];   
  //  MonitorElement* bx_number;
  //  MonitorElement* dbx_chip;
  //  MonitorElement* bx_bmtf_omtf;
  //  MonitorElement* bx_bmtf_emtf;
  //  MonitorElement* bx_omtf_emtfc;


  //  MonitorElement* subs_pt[4];
  //  MonitorElement* subs_qty[4];
  //  MonitorElement* subs_bits[4];

  //  MonitorElement* regional_triggers;
  
  //  MonitorElement* n_bmtf_vs_omtf ;
  //  MonitorElement* n_bmtf_vs_emtf;
  //  MonitorElement* n_omtf_vs_emtf;


  //  const edm::EDGetTokenT<MuonBxCollection> mgmtSource_ ;
  
  /* int bxnum_old_; // bx of previous event */
  /* int obnum_old_; // orbit of previous event */
  /* int trsrc_old_; // code of trigger source ( bits: 0 DT, 1 bRPC, 2 CSC, 3 fRPC ) */

  /* static const double piconv_; */
  /* double phiconv_(float phi); */
  /* void book_(const edm::EventSetup& c); */


};

#endif
