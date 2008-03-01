#ifndef L1TGMT_H
#define L1TGMT_H

/*
 * \file L1TGMT.h
 *
 * $Date: 2007/11/05 09:13:25 $
 * $Revision: 1.4 $
 * \author J. Berryhill, I. Mikulec
 *
*/

// system include files
#include <memory>
#include <unistd.h>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTExtendedCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"

#include <iostream>
#include <fstream>
#include <vector>

//
// class decleration
//

class L1TGMT : public edm::EDAnalyzer {

public:

// Constructor
L1TGMT(const edm::ParameterSet& ps);

// Destructor
virtual ~L1TGMT();

protected:
// Analyze
void analyze(const edm::Event& e, const edm::EventSetup& c);

// BeginJob
void beginJob(const edm::EventSetup& c);

// EndJob
void endJob(void);

private:
  // ----------member data ---------------------------
  DQMStore * dbe;

  MonitorElement* dttf_nbx;
  MonitorElement* dttf_eta;
  MonitorElement* dttf_phi;
  MonitorElement* dttf_ptr;
  MonitorElement* dttf_qty;
  MonitorElement* dttf_etaphi;
  MonitorElement* dttf_bits;
  
  MonitorElement* csctf_nbx;
  MonitorElement* csctf_eta;
  MonitorElement* csctf_phi;
  MonitorElement* csctf_ptr;
  MonitorElement* csctf_qty;
  MonitorElement* csctf_etaphi;
  MonitorElement* csctf_bits;
  
  MonitorElement* rpcb_nbx;
  MonitorElement* rpcb_eta;
  MonitorElement* rpcb_phi;
  MonitorElement* rpcb_ptr;
  MonitorElement* rpcb_qty;
  MonitorElement* rpcb_etaphi;
  MonitorElement* rpcb_bits;
  
  MonitorElement* rpcf_nbx;
  MonitorElement* rpcf_eta;
  MonitorElement* rpcf_phi;
  MonitorElement* rpcf_ptr;
  MonitorElement* rpcf_qty;
  MonitorElement* rpcf_etaphi;
  MonitorElement* rpcf_bits;
  
  MonitorElement* gmt_nbx;
  MonitorElement* gmt_eta;
  MonitorElement* gmt_phi;
  MonitorElement* gmt_ptr;
  MonitorElement* gmt_qty;
  MonitorElement* gmt_etaphi;
  MonitorElement* gmt_bits;

  MonitorElement* n_rpcb_vs_dttf ;
  MonitorElement* n_rpcf_vs_csctf;
  MonitorElement* n_csctf_vs_dttf;
  
  MonitorElement* dttf_dbx;  
  MonitorElement* csctf_dbx;  
  MonitorElement* rpcb_dbx;  
  MonitorElement* rpcf_dbx;
  
  int nev_; // Number of events processed
  std::string outputFile_; //file name for ROOT ouput
  bool verbose_;
  bool monitorDaemon_;
  ofstream logFile_;
  edm::InputTag gmtSource_ ;
  
  int evnum_old_; // event number of previous event
  int bxnum_old_; // bx of previous event
  int trsrc_old_; // code of trigger source ( bits: 0 DT, 1 bRPC, 2 CSC, 3 fRPC )
};

#endif
