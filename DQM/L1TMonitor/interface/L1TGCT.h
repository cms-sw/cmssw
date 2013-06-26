// -*-C++-*-
#ifndef L1TGCT_H
#define L1TGCT_H

/*
 * \file L1TGCT.h
 *
 * $Date: 2012/04/04 09:56:36 $
 * $Revision: 1.25 $
 * \author J. Berryhill
 * $Id: L1TGCT.h,v 1.25 2012/04/04 09:56:36 ghete Exp $
 * $Log: L1TGCT.h,v $
 * Revision 1.25  2012/04/04 09:56:36  ghete
 * Clean up L1TDEMON, add TriggerType hist to RCT, GCT, enable correlation condition tests in GT, clean up HCAL files.
 *
 * Revision 1.24  2012/03/29 21:16:48  rovere
 * Removed all instances of hltTriggerTypeFilter from L1T DQM Code.
 *
 * Revision 1.23  2010/05/30 10:01:58  tapper
 * Added one histogram, correlation of sum ET and HT and changed a few labels for the better.
 *
 * Revision 1.22  2009/11/19 14:33:13  puigh
 * modify beginJob
 *
 * Revision 1.21  2009/11/02 17:00:04  tapper
 * Changes to L1TdeGCT (to include energy sums), to L1TDEMON (should not make any difference now) and L1TGCT to add multiple BXs.
 *
 * Revision 1.20  2009/06/23 09:48:55  tapper
 * Added missing occupancy plot for central and forward jets.
 *
 * Revision 1.19  2009/06/22 15:58:20  tapper
 * Added MET vs MHT correlation plots (both for magnitude and phi). Still untested!
 *
 * Revision 1.18  2009/06/22 15:47:04  tapper
 * Removed rank difference histograms and added MHT. Untested so far!
 *
 * Revision 1.17  2009/05/27 21:49:26  jad
 * updated Total and Missing Energy histograms and added Overlow plots
 *
 * Revision 1.16  2008/11/11 13:20:31  tapper
 * A whole list of house keeping:
 * 1. New shifter histogram with central and forward jets together.
 * 2. Relabelled Ring 0 and Ring 1 to Ring 1 and Ring 2 for HF rings.
 * 3. Tidied up some histograms names to make all consistent.
 * 4. Switched eta and phi in 2D plots to match RCT.
 * 5. Removed 1D eta and phi plots. Will not be needed for Qtests in future.
 *
 * Revision 1.15  2008/09/21 14:33:12  jad
 * updated HF Sums & Counts and added individual Jet Candidates and differences
 *
 * Revision 1.14  2008/06/09 11:08:05  tapper
 * Removed electron sub-folders with histograms per eta and phi bin.
 *
 * Revision 1.13  2008/06/02 11:08:58  tapper
 * Added HF ring histograms....
 *
 * Revision 1.12  2008/04/28 09:23:07  tapper
 * Added 1D eta and phi histograms for electrons and jets as input to Q tests.
 *
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
 void beginJob(void);

// EndJob
void endJob(void);

private:
  // ----------member data ---------------------------
  DQMStore * dbe;

  // trigger type information
  MonitorElement *triggerType_;

  // Jet and EM stuff
  MonitorElement* l1GctAllJetsEtEtaPhi_; 
  MonitorElement* l1GctCenJetsEtEtaPhi_; 
  MonitorElement* l1GctForJetsEtEtaPhi_;
  MonitorElement* l1GctTauJetsEtEtaPhi_;
  MonitorElement* l1GctIsoEmRankEtaPhi_;
  MonitorElement* l1GctNonIsoEmRankEtaPhi_;

  MonitorElement* l1GctAllJetsOccEtaPhi_; 
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

  MonitorElement* l1GctAllJetsOccRankBx_;
  MonitorElement* l1GctAllEmOccRankBx_;

  // Energy sums stuff
  MonitorElement* l1GctEtMiss_;
  MonitorElement* l1GctEtMissPhi_;
  MonitorElement* l1GctEtMissOf_;
  MonitorElement* l1GctEtMissOccBx_;
  MonitorElement* l1GctHtMiss_;
  MonitorElement* l1GctHtMissPhi_;
  MonitorElement* l1GctHtMissOf_;
  MonitorElement* l1GctHtMissOccBx_;
  MonitorElement* l1GctEtMissHtMissCorr_;
  MonitorElement* l1GctEtMissHtMissCorrPhi_;
  MonitorElement* l1GctEtTotal_;
  MonitorElement* l1GctEtTotalOf_;
  MonitorElement* l1GctEtTotalOccBx_;
  MonitorElement* l1GctEtHad_;
  MonitorElement* l1GctEtHadOf_;
  MonitorElement* l1GctEtHadOccBx_;
  MonitorElement* l1GctEtTotalEtHadCorr_;
  
  // HF Rings stuff
  MonitorElement* l1GctHFRing1PosEtaNegEta_;
  MonitorElement* l1GctHFRing2PosEtaNegEta_;
  MonitorElement* l1GctHFRing1TowerCountPosEtaNegEta_;
  MonitorElement* l1GctHFRing2TowerCountPosEtaNegEta_;
  MonitorElement* l1GctHFRing1TowerCountPosEta_;
  MonitorElement* l1GctHFRing1TowerCountNegEta_;
  MonitorElement* l1GctHFRing2TowerCountPosEta_;
  MonitorElement* l1GctHFRing2TowerCountNegEta_;
  MonitorElement* l1GctHFRing1ETSumPosEta_;
  MonitorElement* l1GctHFRing1ETSumNegEta_;
  MonitorElement* l1GctHFRing2ETSumPosEta_;
  MonitorElement* l1GctHFRing2ETSumNegEta_;
  MonitorElement* l1GctHFRingRatioPosEta_;
  MonitorElement* l1GctHFRingRatioNegEta_;
  MonitorElement* l1GctHFRingETSumOccBx_;
  MonitorElement* l1GctHFRingTowerCountOccBx_;

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

  /// filter TriggerType
  int filterTriggerType_;
};

#endif
