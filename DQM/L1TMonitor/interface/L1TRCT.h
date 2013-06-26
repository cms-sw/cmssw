// -*-C++-*-
#ifndef L1TRCT_H
#define L1TRCT_H

/*
 * \file L1TRCT.h
 *
 * $Date: 2012/04/04 09:56:36 $
 * $Revision: 1.9 $
 * \author P. Wittich
 * $Id: L1TRCT.h,v 1.9 2012/04/04 09:56:36 ghete Exp $
 * $Log: L1TRCT.h,v $
 * Revision 1.9  2012/04/04 09:56:36  ghete
 * Clean up L1TDEMON, add TriggerType hist to RCT, GCT, enable correlation condition tests in GT, clean up HCAL files.
 *
 * Revision 1.8  2012/03/29 21:16:48  rovere
 * Removed all instances of hltTriggerTypeFilter from L1T DQM Code.
 *
 * Revision 1.7  2009/11/19 14:34:14  puigh
 * modify beginJob
 *
 * Revision 1.6  2008/11/08 08:45:42  asavin
 * changing the fine grain to HfPlusTau
 *
 * Revision 1.5  2008/07/02 16:53:20  asavin
 * new L1TRCT.h
 *
 * Revision 1.4  2008/03/01 00:40:00  lat
 * DQM core migration.
 *
 * Revision 1.3  2007/09/03 15:14:42  wittich
 * updated RCT with more diagnostic and local coord histos
 *
 * Revision 1.2  2007/02/23 21:58:43  wittich
 * change getByType to getByLabel and add InputTag
 *
 * Revision 1.1  2007/02/19 22:49:53  wittich
 * - Add RCT monitor
 *
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
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// DQM
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"


// Trigger Headers



//
// class declaration
//

class L1TRCT : public edm::EDAnalyzer {

public:

// Constructor
  L1TRCT(const edm::ParameterSet& ps);

// Destructor
 virtual ~L1TRCT();

protected:
// Analyze
 void analyze(const edm::Event& e, const edm::EventSetup& c);

// BeginJob
 void beginJob(void);

// EndJob
void endJob(void);

private:
  // ----------member data ---------------------------
  DQMStore * dbe;

  // trigger type information
  MonitorElement *triggerType_;

  // region global coordinates
  MonitorElement* rctRegionsEtEtaPhi_;
  MonitorElement* rctRegionsOccEtaPhi_;

  // region local coordinates
  MonitorElement* rctRegionsLocalEtEtaPhi_;
  MonitorElement* rctRegionsLocalOccEtaPhi_;
  MonitorElement* rctTauVetoLocalEtaPhi_;

  // Region rank
  MonitorElement* rctRegionRank_;


  MonitorElement* rctOverFlowEtaPhi_;
  MonitorElement* rctTauVetoEtaPhi_;
  MonitorElement* rctMipEtaPhi_;
  MonitorElement* rctQuietEtaPhi_;
  MonitorElement* rctHfPlusTauEtaPhi_;

  // Bx
  MonitorElement *rctRegionBx_;
  MonitorElement *rctEmBx_;

  // em
  // HW coordinates
  MonitorElement *rctEmCardRegion_;


  MonitorElement* rctIsoEmEtEtaPhi_;
  MonitorElement* rctIsoEmOccEtaPhi_;
  MonitorElement* rctNonIsoEmEtEtaPhi_;
  MonitorElement* rctNonIsoEmOccEtaPhi_;
  MonitorElement* rctIsoEmRank_;
  MonitorElement* rctNonIsoEmRank_;


  int nev_; // Number of events processed
  std::string outputFile_; //file name for ROOT ouput
  bool verbose_;
  bool monitorDaemon_;
  ofstream logFile_;

  edm::InputTag rctSource_;

  /// filter TriggerType
  int filterTriggerType_;

};

#endif
