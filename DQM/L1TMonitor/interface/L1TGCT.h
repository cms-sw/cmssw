// -*-C++-*-
#ifndef L1TGCT_H
#define L1TGCT_H

/*
 * \file L1TGCT.h
 *
 * $Date: 2008/04/25 15:40:21 $
 * $Revision: 1.11 $
 * \author J. Berryhill
 * $Id: L1TGCT.h,v 1.11 2008/04/25 15:40:21 tapper Exp $
 * $Log: L1TGCT.h,v $
 * Revision 1.11  2008/04/25 15:40:21  tapper
 * Added histograms to EventInfo//errorSummarySegments.
 *
 * Revision 1.10  2008/03/01 00:40:00  lat
 * DQM core migration.
 *
 * Revision 1.9  2008/02/20 19:24:24  tapper
 * Removed noisy include.
 *
 * Revision 1.8  2008/02/20 18:59:29  tapper
 * Ported GCTMonitor histograms into L1TGCT
 *
 * Revision 1.7  2007/09/04 02:54:21  wittich
 * - fix dupe ME in RCT
 * - put in rank>0 req in GCT
 * - various small other fixes
 *
 * Revision 1.6  2007/08/31 18:14:20  wittich
 * update GCT packages to reflect GctRawToDigi, and move to raw plots
 *
 * Revision 1.5  2007/08/31 11:02:55  wittich
 * cerr -> LogInfo
 *
 * Revision 1.4  2007/02/22 19:43:52  berryhil
 *
 *
 *
 * InputTag parameters added for all modules
 *
 * Revision 1.3  2007/02/19 22:49:53  wittich
 * - Add RCT monitor
 *
 * Revision 1.2  2007/02/19 21:11:23  wittich
 * - Updates for integrating GCT monitor.
 *   + Adapted right now only the L1E elements thereof.
 *   + added DataFormats/L1Trigger to build file.
 *
 *
*/

// system include files
#include <memory>
#include <unistd.h>


#include <iostream>
#include <fstream>
#include <vector>


// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

// DQM
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"





//
// class declaration
//

class L1TGCT : public edm::EDAnalyzer {

public:

// Constructor
  L1TGCT(const edm::ParameterSet& ps);

// Destructor
 virtual ~L1TGCT();

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

  // Summary stuff
  MonitorElement* l1GctSummIsoEmRankEtaPhi_;
  MonitorElement* l1GctSummNonIsoEmRankEtaPhi_;

  // GCT stuff
  MonitorElement* l1GctCenJetsEtEtaPhi_; 
  MonitorElement* l1GctForJetsEtEtaPhi_;
  MonitorElement* l1GctTauJetsEtEtaPhi_;
  MonitorElement* l1GctIsoEmRankEtaPhi_;
  MonitorElement* l1GctNonIsoEmRankEtaPhi_;

  MonitorElement* l1GctCenJetsOccEtaPhi_;
  MonitorElement* l1GctForJetsOccEtaPhi_;  
  MonitorElement* l1GctTauJetsOccEtaPhi_;  
  MonitorElement* l1GctIsoEmOccEtaPhi_;    
  MonitorElement* l1GctNonIsoEmOccEtaPhi_; 

  MonitorElement* l1GctCenJetsOccEta_;
  MonitorElement* l1GctForJetsOccEta_;  
  MonitorElement* l1GctTauJetsOccEta_;  
  MonitorElement* l1GctIsoEmOccEta_;    
  MonitorElement* l1GctNonIsoEmOccEta_; 

  MonitorElement* l1GctCenJetsOccPhi_;
  MonitorElement* l1GctForJetsOccPhi_;  
  MonitorElement* l1GctTauJetsOccPhi_;  
  MonitorElement* l1GctIsoEmOccPhi_;    
  MonitorElement* l1GctNonIsoEmOccPhi_; 

  MonitorElement* l1GctCenJetsRank_;
  MonitorElement* l1GctForJetsRank_;
  MonitorElement* l1GctTauJetsRank_;
  MonitorElement* l1GctIsoEmRank_;
  MonitorElement* l1GctNonIsoEmRank_;

  MonitorElement* l1GctEtMiss_;
  MonitorElement* l1GctEtMissPhi_;
  MonitorElement* l1GctEtTotal_;
  MonitorElement* l1GctEtHad_;

  // GCT electron stuff
  MonitorElement* l1GctIsoEmRankBin_[22][18];
  MonitorElement* l1GctNonIsoEmRankBin_[22][18];

  MonitorElement* l1GctIsoEmRankCand0_;
  MonitorElement* l1GctIsoEmRankCand1_;
  MonitorElement* l1GctIsoEmRankCand2_;
  MonitorElement* l1GctIsoEmRankCand3_;

  MonitorElement* l1GctNonIsoEmRankCand0_;
  MonitorElement* l1GctNonIsoEmRankCand1_;
  MonitorElement* l1GctNonIsoEmRankCand2_;
  MonitorElement* l1GctNonIsoEmRankCand3_;

  MonitorElement* l1GctIsoEmRankDiff01_;
  MonitorElement* l1GctIsoEmRankDiff12_;
  MonitorElement* l1GctIsoEmRankDiff23_;
  MonitorElement* l1GctNonIsoEmRankDiff01_;
  MonitorElement* l1GctNonIsoEmRankDiff12_;
  MonitorElement* l1GctNonIsoEmRankDiff23_;

  int nev_; // Number of events processed
  std::string outputFile_; //file name for ROOT ouput
  bool verbose_;
  bool monitorDaemon_;
  ofstream logFile_;

  edm::InputTag gctCenJetsSource_;
  edm::InputTag gctForJetsSource_;
  edm::InputTag gctTauJetsSource_;
  edm::InputTag gctEnergySumsSource_;
  edm::InputTag gctIsoEmSource_;
  edm::InputTag gctNonIsoEmSource_;

};

#endif
