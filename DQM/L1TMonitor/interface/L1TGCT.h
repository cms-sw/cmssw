// -*-C++-*-
#ifndef L1TGCT_H
#define L1TGCT_H

/*
 * \file L1TGCT.h
 *
 * $Date: 2007/08/31 18:14:20 $
 * $Revision: 1.6 $
 * \author J. Berryhill
 * $Id: L1TGCT.h,v 1.6 2007/08/31 18:14:20 wittich Exp $
 * $Log: L1TGCT.h,v $
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
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"





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
  DaqMonitorBEInterface * dbe;

  // L1Extra stuff
  MonitorElement* l1GctCenJetsEtEtaPhi_; 
  MonitorElement* l1GctForJetsEtEtaPhi_;
  MonitorElement* l1GctTauJetsEtEtaPhi_;
  MonitorElement* l1GctIsoEmEtEtaPhi_;
  MonitorElement* l1GctNonIsoEmEtEtaPhi_;

  MonitorElement* l1GctCenJetsOccEtaPhi_;
  MonitorElement* l1GctForJetsOccEtaPhi_;  
  MonitorElement* l1GctTauJetsOccEtaPhi_;  
  MonitorElement* l1GctIsoEmOccEtaPhi_;    
  MonitorElement* l1GctNonIsoEmOccEtaPhi_; 

  MonitorElement* l1GctCenJetsRank_;
  MonitorElement* l1GctForJetsRank_;
  MonitorElement* l1GctTauJetsRank_;
  MonitorElement* l1GctIsoEmRank_;
  MonitorElement* l1GctNonIsoEmRank_;

  MonitorElement* l1GctEtMiss_;
  MonitorElement* l1GctEtMissPhi_;

  // AFAIK, these don't have phi values
  MonitorElement* l1GctEtTotal_;
  MonitorElement* l1GctEtHad_;


  int nev_; // Number of events processed
  std::string outputFile_; //file name for ROOT ouput
  bool verbose_;
  bool monitorDaemon_;
  ofstream logFile_;

  edm::InputTag gctSource_;

};

#endif
