#ifndef L1TGMT_H
#define L1TGMT_H

/*
 * \file L1TGMT.h
 *
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
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTExtendedCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include <iostream>
#include <fstream>
#include <vector>

//
// class decleration
//

class L1TGMT : public DQMEDAnalyzer {
public:
  // Constructor
  L1TGMT(const edm::ParameterSet& ps);

  // Destructor
  ~L1TGMT() override;

protected:
  // Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  void bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const&) override;

private:
  // ----------member data ---------------------------

  enum ensubs { DTTF = 0, RPCb, CSCTF, RPCf, GMT };

  MonitorElement* subs_nbx[5];
  MonitorElement* subs_eta[5];
  MonitorElement* subs_phi[5];
  MonitorElement* subs_pt[5];
  MonitorElement* subs_qty[5];
  MonitorElement* subs_etaphi[5];
  MonitorElement* subs_etaqty[5];
  MonitorElement* subs_bits[5];

  MonitorElement* regional_triggers;

  MonitorElement* bx_number;
  MonitorElement* dbx_chip;
  MonitorElement* eta_dtcsc_and_rpc;
  MonitorElement* eta_dtcsc_only;
  MonitorElement* eta_rpc_only;
  MonitorElement* phi_dtcsc_and_rpc;
  MonitorElement* phi_dtcsc_only;
  MonitorElement* phi_rpc_only;
  MonitorElement* etaphi_dtcsc_and_rpc;
  MonitorElement* etaphi_dtcsc_only;
  MonitorElement* etaphi_rpc_only;
  MonitorElement* dist_phi_dt_rpc;
  MonitorElement* dist_phi_csc_rpc;
  MonitorElement* dist_phi_dt_csc;
  MonitorElement* dist_eta_dt_rpc;
  MonitorElement* dist_eta_csc_rpc;
  MonitorElement* dist_eta_dt_csc;
  MonitorElement* bx_dt_rpc;
  MonitorElement* bx_csc_rpc;
  MonitorElement* bx_dt_csc;

  MonitorElement* n_rpcb_vs_dttf;
  MonitorElement* n_rpcf_vs_csctf;
  MonitorElement* n_csctf_vs_dttf;

  MonitorElement* subs_dbx[4];

  const bool verbose_;
  std::ofstream logFile_;
  const edm::EDGetTokenT<L1MuGMTReadoutCollection> gmtSource_;

  int bxnum_old_;  // bx of previous event
  int obnum_old_;  // orbit of previous event
  int trsrc_old_;  // code of trigger source ( bits: 0 DT, 1 bRPC, 2 CSC, 3 fRPC )

  static const double piconv_;
  double phiconv_(float phi);
  void book_(const edm::EventSetup& c);
};

#endif
