#ifndef L1TRPCTF_H
#define L1TRPCTF_H

/*
 * \file L1TRPCTF.h
 *
 * \author J. Berryhill
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
#include <set>

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
//
// class decleration
//

class L1TRPCTF : public DQMEDAnalyzer {

public:

// Constructor
 L1TRPCTF(const edm::ParameterSet& ps);

// Destructor
 virtual ~L1TRPCTF();

protected:
// Analyze
 void analyze(const edm::Event& e, const edm::EventSetup& c);

// BeginJob
  virtual void bookHistograms(DQMStore::IBooker &ibooker, const edm::Run&, const edm::EventSetup&) override;
  virtual void dqmBeginRun(const edm::Run&, const edm::EventSetup&);

 virtual void beginLuminosityBlock(const edm::LuminosityBlock& l, const edm::EventSetup& c);
 void endLuminosityBlock(const edm::LuminosityBlock& l, const edm::EventSetup& c);


private:

  
  // ----------member data ---------------------------

  MonitorElement* rpctfetavalue[3];
  MonitorElement* rpctfphivalue[3];
  MonitorElement* rpctfptvalue[3];
  MonitorElement* rpctfchargevalue[3];
  MonitorElement* rpctfquality[3];
  MonitorElement* rpctfntrack_b[3];
  MonitorElement* rpctfntrack_e[3];
  MonitorElement* rpctfbx;
  MonitorElement* m_qualVsEta[3];
  MonitorElement* m_muonsEtaPhi[3];
  //MonitorElement* m_phipacked;

  MonitorElement* m_bxDiff;
  MonitorElement* rpctfcratesynchro[12];

  std::set<unsigned long long int>  m_globBX;

  edm::EDGetTokenT<L1MuGMTReadoutCollection> rpctfSource_ ;

  int nev_; // Number of events processed
  int nevRPC_; // Number of events processed where muon was found by rpc trigger
  std::string outputFile_; //file name for ROOT ouput
  bool verbose_;
  bool monitorDaemon_;
  //bool m_rpcDigiFine;
  //bool m_useRpcDigi;

  long long int m_lastUsedBxInBxdiff;
  std::string output_dir_;
  struct BxDelays { int bx, eta_t, phi_p; };  


};

#endif
