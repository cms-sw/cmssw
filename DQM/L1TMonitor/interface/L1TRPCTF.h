#ifndef L1TRPCTF_H
#define L1TRPCTF_H

/*
 * \file L1TRPCTF.h
 *
 * $Date: 2011/06/02 09:32:54 $
 * $Revision: 1.22 $
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

//
// class decleration
//

class L1TRPCTF : public edm::EDAnalyzer {

public:

// Constructor
L1TRPCTF(const edm::ParameterSet& ps);

// Destructor
virtual ~L1TRPCTF();

protected:
// Analyze
void analyze(const edm::Event& e, const edm::EventSetup& c);

// BeginJob
void beginJob(void);

// EndJob
void endJob(void);

void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                          const edm::EventSetup& context);
void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                        const edm::EventSetup& c);
                        
void endRun(const edm::Run & r, const edm::EventSetup & c);


private:

  
  // ----------member data ---------------------------
  DQMStore * m_dbe;

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
  
  


  edm::InputTag rpctfSource_ ;

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
