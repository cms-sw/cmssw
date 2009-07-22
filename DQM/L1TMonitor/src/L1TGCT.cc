/*
 * \file L1TGCT.cc
 *
 * $Date: 2009/06/23 09:48:55 $
 * $Revision: 1.42 $
 * \author J. Berryhill
 *
 * $Log: L1TGCT.cc,v $
 * Revision 1.42  2009/06/23 09:48:55  tapper
 * Added missing occupancy plot for central and forward jets.
 *
 * Revision 1.41  2009/06/22 15:58:20  tapper
 * Added MET vs MHT correlation plots (both for magnitude and phi). Still untested!
 *
 * Revision 1.39  2009/05/27 21:49:26  jad
 * updated Total and Missing Energy histograms and added Overlow plots
 *
 * Revision 1.38  2009/02/24 13:01:42  jad
 * Updated MET_PHI histogram to obey the correct limits
 *
 * Revision 1.37  2008/11/11 13:20:32  tapper
 * A whole list of house keeping:
 * 1. New shifter histogram with central and forward jets together.
 * 2. Relabelled Ring 0 and Ring 1 to Ring 1 and Ring 2 for HF rings.
 * 3. Tidied up some histograms names to make all consistent.
 * 4. Switched eta and phi in 2D plots to match RCT.
 * 5. Removed 1D eta and phi plots. Will not be needed for Qtests in future.
 *
 * Revision 1.36  2008/10/28 14:16:16  tapper
 * Tidied up and removed some unnecessary code.
 *
 * Revision 1.35  2008/10/24 08:38:54  jbrooke
 * fix empty jet plots
 *
 * Revision 1.34  2008/10/10 12:41:24  jbrooke
 * put back checks on energy sum vector size, change [] to .at()
 *
 * Revision 1.33  2008/09/21 14:37:51  jad
 * updated HF Sums & Counts and added individual Jet Candidates and differences
 *
 * Revision 1.30  2008/06/09 11:07:52  tapper
 * Removed electron sub-folders with histograms per eta and phi bin.
 *
 * Revision 1.29  2008/06/06 15:18:22  tapper
 * Removed errorSummary folder stuff.
 *
 * Revision 1.28  2008/06/02 11:08:58  tapper
 * Added HF ring histograms....
 *
 * Revision 1.27  2008/05/12 12:52:46  tapper
 * Fixed problem when no GCT data in the event.
 *
 * Revision 1.26  2008/05/09 16:42:27  ameyer
 * *** empty log message ***
 *
 * Revision 1.25  2008/04/29 15:24:49  tapper
 * Changed path to summary histograms.
 *
 * Revision 1.24  2008/04/28 09:23:07  tapper
 * Added 1D eta and phi histograms for electrons and jets as input to Q tests.
 *
 * Revision 1.23  2008/04/25 15:40:21  tapper
 * Added histograms to EventInfo//errorSummarySegments.
 *
 * Revision 1.22  2008/03/20 19:38:25  berryhil
 *
 *
 * organized message logger
 *
 * Revision 1.21  2008/03/14 20:35:46  berryhil
 *
 *
 * stripped out obsolete parameter settings
 *
 * rpc tpg restored with correct dn access and dbe handling
 *
 * Revision 1.20  2008/03/12 17:24:24  berryhil
 *
 *
 * eliminated log files, truncated HCALTPGXana histo output
 *
 * Revision 1.19  2008/03/01 00:40:00  lat
 * DQM core migration.
 *
 * Revision 1.18  2008/02/20 18:59:29  tapper
 * Ported GCTMonitor histograms into L1TGCT
 *
 * Revision 1.17  2008/01/22 18:56:02  muzaffar
 * include cleanup. Only for cc/cpp files
 *
 * Revision 1.16  2007/12/21 17:41:20  berryhil
 *
 *
 * try/catch removal
 *
 * Revision 1.15  2007/11/19 15:08:22  lorenzo
 * changed top folder name
 *
 * Revision 1.14  2007/09/27 23:01:28  ratnik
 * QA campaign: fixes to compensate includes cleanup in  DataFormats/L1Trigger
 *
 * Revision 1.13  2007/09/27 16:56:26  wittich
 * verbosity fixes
 *
 * Revision 1.12  2007/09/26 15:26:23  berryhil
 *
 *
 * restored L1TGCT.cc
 *
 * Revision 1.10  2007/09/05 22:31:36  wittich
 * - Factorize getByLabels to approximate my understanding of what the
 *   HW can do.
 * - tested (loosely speaking) on GREJ' data.
 *
 * Revision 1.9  2007/09/04 02:54:19  wittich
 * - fix dupe ME in RCT
 * - put in rank>0 req in GCT
 * - various small other fixes
 *
 * Revision 1.8  2007/08/31 18:14:21  wittich
 * update GCT packages to reflect GctRawToDigi, and move to raw plots
 *
 * Revision 1.7  2007/08/31 11:02:56  wittich
 * cerr -> LogInfo
 *
 * Revision 1.6  2007/02/22 19:43:53  berryhil
 *
 *
 *
 * InputTag parameters added for all modules
 *
 *
 *
 */

#include "DQM/L1TMonitor/interface/L1TGCT.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Trigger Headers

// GCT and RCT data formats
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtSums.h"
#include "DQMServices/Core/interface/DQMStore.h"

using namespace edm;

// Define statics for bins etc.
const unsigned int ETABINS = 22;
const float ETAMIN = -0.5;
const float ETAMAX = 21.5;

const unsigned int METPHIBINS = 72;
const float METPHIMIN = -0.5;
const float METPHIMAX = 71.5;

const unsigned int PHIBINS = 18;
const float PHIMIN = -0.5;
const float PHIMAX = 17.5;

const unsigned int OFBINS = 2;
const float OFMIN = -0.5;
const float OFMAX = 1.5;

// Bins for 3, 5, 6, 7, 10 and 12 bits
const unsigned int R3BINS = 8;
const float R3MIN = -0.5;
const float R3MAX = 7.5;
const unsigned int R5BINS = 32;
const float R5MIN = -0.5;
const float R5MAX = 31.5;
const unsigned int R6BINS = 64;
const float R6MIN = -0.5;
const float R6MAX = 63.5;
const unsigned int R7BINS = 128;
const float R7MIN = -0.5;
const float R7MAX = 127.5;
const unsigned int R10BINS = 1024;
const float R10MIN = -0.5;
const float R10MAX = 1023.5;
const unsigned int R12BINS = 4096;
const float R12MIN = -0.5;
const float R12MAX = 4095.5;

L1TGCT::L1TGCT(const edm::ParameterSet & ps) :
  gctCenJetsSource_(ps.getParameter<edm::InputTag>("gctCentralJetsSource")),
  gctForJetsSource_(ps.getParameter<edm::InputTag>("gctForwardJetsSource")),
  gctTauJetsSource_(ps.getParameter<edm::InputTag>("gctTauJetsSource")),
  gctEnergySumsSource_(ps.getParameter<edm::InputTag>("gctEnergySumsSource")),
  gctIsoEmSource_(ps.getParameter<edm::InputTag>("gctIsoEmSource")),
  gctNonIsoEmSource_(ps.getParameter<edm::InputTag>("gctNonIsoEmSource"))
{

  // verbosity switch
  verbose_ = ps.getUntrackedParameter < bool > ("verbose", false);

  if (verbose_)
    edm::LogInfo("L1TGCT") << "L1TGCT: constructor...." << std::endl;


  dbe = NULL;
  if (ps.getUntrackedParameter < bool > ("DQMStore", false)) {
    dbe = edm::Service < DQMStore > ().operator->();
    dbe->setVerbose(0);
  }

  outputFile_ = ps.getUntrackedParameter < std::string > ("outputFile", "");
  if (outputFile_.size() != 0) {
    edm::LogInfo("L1TGCT") << "L1T Monitoring histograms will be saved to "
                           << outputFile_ << std::endl;
  }

  bool disable = ps.getUntrackedParameter<bool>("disableROOToutput", false);
  if(disable){
    outputFile_="";
  }


  if (dbe != NULL) {
    dbe->setCurrentFolder("L1T/L1TGCT");
  }

}

L1TGCT::~L1TGCT()
{
}

void L1TGCT::beginJob(const edm::EventSetup & c)
{

  nev_ = 0;

  // get hold of back-end interface
  DQMStore *dbe = 0;
  dbe = edm::Service < DQMStore > ().operator->();

  if (dbe) {
    dbe->setCurrentFolder("L1T/L1TGCT");
    dbe->rmdir("L1T/L1TGCT");
  }


  if (dbe) {

    dbe->setCurrentFolder("L1T/L1TGCT");

    l1GctAllJetsEtEtaPhi_ = dbe->book2D("AllJetsEtEtaPhi", "CENTRAL AND FORWARD JET RANK",
					ETABINS, ETAMIN, ETAMAX,
                                        PHIBINS, PHIMIN, PHIMAX);
    l1GctCenJetsEtEtaPhi_ = dbe->book2D("CenJetsEtEtaPhi", "CENTRAL JET RANK",
					ETABINS, ETAMIN, ETAMAX,
                                        PHIBINS, PHIMIN, PHIMAX); 
    l1GctForJetsEtEtaPhi_ = dbe->book2D("ForJetsEtEtaPhi", "FORWARD JET RANK",
					ETABINS, ETAMIN, ETAMAX,
					PHIBINS, PHIMIN, PHIMAX); 
    l1GctTauJetsEtEtaPhi_ = dbe->book2D("TauJetsEtEtaPhi", "TAU JET RANK", 
					14, 3.5, 17.5,
					PHIBINS, PHIMIN, PHIMAX); 
    l1GctIsoEmRankEtaPhi_ = dbe->book2D("IsoEmRankEtaPhi", "ISO EM RANK", 
                                        14, 3.5, 17.5,
                                        PHIBINS, PHIMIN, PHIMAX); 		    
    l1GctNonIsoEmRankEtaPhi_ = dbe->book2D("NonIsoEmRankEtaPhi", "NON-ISO EM RANK",
                                           14, 3.5, 17.5,
                                           PHIBINS, PHIMIN, PHIMAX); 
    l1GctAllJetsOccEtaPhi_ = dbe->book2D("AllJetsOccEtaPhi", "CENTRAL AND FORWARD JET OCCUPANCY",
					ETABINS, ETAMIN, ETAMAX,
                                        PHIBINS, PHIMIN, PHIMAX);
    l1GctCenJetsOccEtaPhi_ = dbe->book2D("CenJetsOccEtaPhi", "CENTRAL JET OCCUPANCY",
					 ETABINS, ETAMIN, ETAMAX,
                                         PHIBINS, PHIMIN, PHIMAX); 
    l1GctForJetsOccEtaPhi_ = dbe->book2D("ForJetsOccEtaPhi", "FORWARD JET OCCUPANCY",
					 ETABINS, ETAMIN, ETAMAX,
					 PHIBINS, PHIMIN, PHIMAX);
    l1GctTauJetsOccEtaPhi_ = dbe->book2D("TauJetsOccEtaPhi", "TAU JET OCCUPANCY",
					 14, 3.5, 17.5,
					 PHIBINS, PHIMIN, PHIMAX); 
    l1GctIsoEmOccEtaPhi_ = dbe->book2D("IsoEmOccEtaPhi", "ISO EM OCCUPANCY",
				       14, 3.5, 17.5,
				       PHIBINS, PHIMIN, PHIMAX); 
    l1GctNonIsoEmOccEtaPhi_ = dbe->book2D("NonIsoEmOccEtaPhi", "NON-ISO EM OCCUPANCY",
					  14, 3.5, 17.5,
					  PHIBINS, PHIMIN, PHIMAX); 

    l1GctHFRing1PosEtaNegEta_ = dbe->book2D("HFRing1Corr", "HF RING1 CORRELATION NEG POS ETA",
                                            ETABINS, ETAMIN, ETAMAX,
                                            PHIBINS, PHIMIN, PHIMAX); 
    l1GctHFRing2PosEtaNegEta_ = dbe->book2D("HFRing2Corr", "HF RING2 CORRELATION NEG POS ETA",
                                            ETABINS, ETAMIN, ETAMAX,
                                            PHIBINS, PHIMIN, PHIMAX); 
    l1GctHFRing1TowerCountPosEtaNegEta_ = dbe->book2D("HFRing1TowerCountCorr", "HF RING1 TOWER COUNT CORRELATION NEG POS ETA",
                                                      ETABINS, ETAMIN, ETAMAX,
                                                      PHIBINS, PHIMIN, PHIMAX);
    l1GctHFRing2TowerCountPosEtaNegEta_ = dbe->book2D("HFRing2TowerCountCorr", "HF RING2 TOWER COUNT CORRELATION NEG POS ETA",
                                                      ETABINS, ETAMIN, ETAMAX,
                                                      PHIBINS, PHIMIN, PHIMAX); 

    //HF Ring stuff
    l1GctHFRing1TowerCountPosEta_ = dbe->book1D("HFRing1TowerCountPosEta", "POS ETA RING1 HFRING BIT", R3BINS, R3MIN, R3MAX);
    l1GctHFRing1TowerCountNegEta_ = dbe->book1D("HFRing1TowerCountNegEta", "NEG ETA RING1 HFRING BIT", R3BINS, R3MIN, R3MAX);
    l1GctHFRing2TowerCountPosEta_ = dbe->book1D("HFRing2TowerCountPosEta", "POS ETA RING2 HFRING BIT", R3BINS, R3MIN, R3MAX);
    l1GctHFRing2TowerCountNegEta_ = dbe->book1D("HFRing2TowerCountNegEta", "NEG ETA RING2 HFRING BIT", R3BINS, R3MIN, R3MAX);

    l1GctHFRing1ETSumPosEta_ = dbe->book1D("HFRing1ETSumPosEta", "POS ETA RING1 ET SUM", R3BINS, R3MIN, R3MAX);
    l1GctHFRing1ETSumNegEta_ = dbe->book1D("HFRing1ETSumNegEta", "NEG ETA RING1 ET SUM", R3BINS, R3MIN, R3MAX);
    l1GctHFRing2ETSumPosEta_ = dbe->book1D("HFRing2ETSumPosEta", "POS ETA RING2 ET SUM", R3BINS, R3MIN, R3MAX);
    l1GctHFRing2ETSumNegEta_ = dbe->book1D("HFRing2ETSumNegEta", "NEG ETA RING2 ET SUM", R3BINS, R3MIN, R3MAX);
    l1GctHFRingRatioPosEta_  = dbe->book1D("HFRingRatioPosEta", "RING RATIO POS ETA", R5BINS, R5MIN, R5MAX);
    l1GctHFRingRatioNegEta_  = dbe->book1D("HFRingRatioNegEta", "RING RATIO NEG ETA", R5BINS, R5MIN, R5MAX);
    
    // Rank histograms
    l1GctCenJetsRank_  = dbe->book1D("CenJetsRank", "CENTRAL JET RANK", R6BINS, R6MIN, R6MAX);
    l1GctForJetsRank_  = dbe->book1D("ForJetsRank", "FORWARD JET RANK", R6BINS, R6MIN, R6MAX);
    l1GctTauJetsRank_  = dbe->book1D("TauJetsRank", "TAU JET RANK", R6BINS, R6MIN, R6MAX);
    l1GctIsoEmRank_    = dbe->book1D("IsoEmRank", "ISO EM RANK", R6BINS, R6MIN, R6MAX);
    l1GctNonIsoEmRank_ = dbe->book1D("NonIsoEmRank", "NON-ISO EM RANK", R6BINS, R6MIN, R6MAX);

    // Energy sums
    l1GctEtMiss_    = dbe->book1D("EtMiss", "MISSING E_{T}", R12BINS, R12MIN, R12MAX);
    l1GctEtMissPhi_ = dbe->book1D("EtMissPhi", "MISSING E_{T} #phi", METPHIBINS, METPHIMIN, METPHIMAX);
    l1GctEtMissOf_  = dbe->book1D("EtMissOf", "MISSING E_{T} OVERFLOW", OFBINS, OFMIN, OFMAX);
    l1GctHtMiss_    = dbe->book1D("HtMiss", "MISSING H_{T}", R7BINS, R7MIN, R7MAX);
    l1GctHtMissPhi_ = dbe->book1D("HtMissPhi", "MISSING H_{T} #phi", R5BINS, R5MIN, R5MAX);
    l1GctHtMissOf_  = dbe->book1D("HtMissOf", "MISSING H_{T} OVERFLOW", OFBINS, OFMIN, OFMAX);
    l1GctEtMissHtMissCorr_ = dbe->book2D("EtMissHtMissCorr", "MISSING E_{T} MISSING H_{T} CORRELATION",
                                         R12BINS, R12MIN, R12MAX,
                                         R7BINS, R7MIN, R7MAX); 
    l1GctEtMissHtMissCorrPhi_ = dbe->book2D("EtMissHtMissPhiCorr", "MISSING E_{T} MISSING H_{T} #phi CORRELATION",
                                            METPHIBINS, METPHIMIN, METPHIMAX,
                                            R5BINS, R5MIN, R5MAX);
    l1GctEtTotal_   = dbe->book1D("EtTotal", "TOTAL E_{T}", R12BINS, R12MIN, R12MAX);
    l1GctEtTotalOf_ = dbe->book1D("EtTotalOf", "TOTAL E_{T} OVERFLOW", OFBINS, OFMIN, OFMAX);
    l1GctEtHad_     = dbe->book1D("EtHad", "TOTAL HAD E_{T}", R12BINS, R12MIN, R12MAX);
    l1GctEtHadOf_   = dbe->book1D("EtHadOf", "TOTAL HAD E_{T} OVERFLOW", OFBINS, OFMIN, OFMAX);

    // More detailed quantities
    l1GctIsoEmRankCand0_ = dbe->book1D("IsoEmRankCand0","ISO EM RANK CAND 0", R6BINS, R6MIN, R6MAX);
    l1GctIsoEmRankCand1_ = dbe->book1D("IsoEmRankCand1","ISO EM RANK CAND 1", R6BINS, R6MIN, R6MAX);
    l1GctIsoEmRankCand2_ = dbe->book1D("IsoEmRankCand2","ISO EM RANK CAND 2", R6BINS, R6MIN, R6MAX);
    l1GctIsoEmRankCand3_ = dbe->book1D("IsoEmRankCand3","ISO EM RANK CAND 3", R6BINS, R6MIN, R6MAX);

    l1GctNonIsoEmRankCand0_ = dbe->book1D("NonIsoEmRankCand0","NON-ISO EM RANK CAND 0", R6BINS, R6MIN, R6MAX);
    l1GctNonIsoEmRankCand1_ = dbe->book1D("NonIsoEmRankCand1","NON-ISO EM RANK CAND 1", R6BINS, R6MIN, R6MAX);
    l1GctNonIsoEmRankCand2_ = dbe->book1D("NonIsoEmRankCand2","NON-ISO EM RANK CAND 2", R6BINS, R6MIN, R6MAX);
    l1GctNonIsoEmRankCand3_ = dbe->book1D("NonIsoEmRankCand3","NON-ISO EM RANK CAND 3", R6BINS, R6MIN, R6MAX);

    l1GctCenJetsRankCand0_ = dbe->book1D("CenJetsRankCand0","CEN JET RANK CAND 0", R6BINS, R6MIN, R6MAX);
    l1GctCenJetsRankCand1_ = dbe->book1D("CenJetsRankCand1","CEN JET RANK CAND 1", R6BINS, R6MIN, R6MAX);
    l1GctCenJetsRankCand2_ = dbe->book1D("CenJetsRankCand2","CEN JET RANK CAND 2", R6BINS, R6MIN, R6MAX);
    l1GctCenJetsRankCand3_ = dbe->book1D("CenJetsRankCand3","CEN JET RANK CAND 3", R6BINS, R6MIN, R6MAX);

    l1GctForJetsRankCand0_ = dbe->book1D("ForJetsRankCand0","FOR JET RANK CAND 0", R6BINS, R6MIN, R6MAX);
    l1GctForJetsRankCand1_ = dbe->book1D("ForJetsRankCand1","FOR JET RANK CAND 1", R6BINS, R6MIN, R6MAX);
    l1GctForJetsRankCand2_ = dbe->book1D("ForJetsRankCand2","FOR JET RANK CAND 2", R6BINS, R6MIN, R6MAX);
    l1GctForJetsRankCand3_ = dbe->book1D("ForJetsRankCand3","FOR JET RANK CAND 3", R6BINS, R6MIN, R6MAX);

    l1GctTauJetsRankCand0_ = dbe->book1D("TauJetsRankCand0","TAU JET RANK CAND 0", R6BINS, R6MIN, R6MAX);
    l1GctTauJetsRankCand1_ = dbe->book1D("TauJetsRankCand1","TAU JET RANK CAND 1", R6BINS, R6MIN, R6MAX);
    l1GctTauJetsRankCand2_ = dbe->book1D("TauJetsRankCand2","TAU JET RANK CAND 2", R6BINS, R6MIN, R6MAX);
    l1GctTauJetsRankCand3_ = dbe->book1D("TauJetsRankCand3","TAU JET RANK CAND 3", R6BINS, R6MIN, R6MAX);

  }

}


void L1TGCT::endJob(void)
{
  if (verbose_)
    edm::LogInfo("L1TGCT") << "L1TGCT: end job...." << std::endl;
  edm::LogInfo("EndJob") << "analyzed " << nev_ << " events";

  if (outputFile_.size() != 0 && dbe) {
    dbe->save(outputFile_);
  }

  return;
}

void L1TGCT::analyze(const edm::Event & e, const edm::EventSetup & c)
{
  nev_++;
  if (verbose_) {
    edm::LogInfo("L1TGCT") << "L1TGCT: analyze...." << std::endl;
  }
  
  // Get all the collections
  edm::Handle < L1GctEmCandCollection > l1IsoEm;
  edm::Handle < L1GctEmCandCollection > l1NonIsoEm;
  edm::Handle < L1GctJetCandCollection > l1CenJets;
  edm::Handle < L1GctJetCandCollection > l1ForJets;
  edm::Handle < L1GctJetCandCollection > l1TauJets;
  edm::Handle < L1GctHFRingEtSumsCollection > l1HFSums; 
  edm::Handle < L1GctHFBitCountsCollection > l1HFCounts;
  edm::Handle < L1GctEtMissCollection >  l1EtMiss;
  edm::Handle < L1GctHtMissCollection >  l1HtMiss;
  edm::Handle < L1GctEtHadCollection >   l1EtHad;
  edm::Handle < L1GctEtTotalCollection > l1EtTotal;

  e.getByLabel(gctIsoEmSource_, l1IsoEm);
  e.getByLabel(gctNonIsoEmSource_, l1NonIsoEm);
  e.getByLabel(gctCenJetsSource_, l1CenJets);
  e.getByLabel(gctForJetsSource_, l1ForJets);
  e.getByLabel(gctTauJetsSource_, l1TauJets);
  e.getByLabel(gctEnergySumsSource_, l1HFSums);
  e.getByLabel(gctEnergySumsSource_, l1HFCounts);  
  e.getByLabel(gctEnergySumsSource_, l1EtMiss);
  e.getByLabel(gctEnergySumsSource_, l1HtMiss);
  e.getByLabel(gctEnergySumsSource_, l1EtHad);
  e.getByLabel(gctEnergySumsSource_, l1EtTotal);

  // Fill histograms

  // Central jets
  if (l1CenJets.isValid()) {
    for (L1GctJetCandCollection::const_iterator cj = l1CenJets->begin();cj != l1CenJets->end(); cj++) {
      l1GctCenJetsRank_->Fill(cj->rank());
      // only plot eta and phi maps for non-zero candidates
      if (cj->rank()) {
        l1GctAllJetsEtEtaPhi_->Fill(cj->regionId().ieta(),cj->regionId().iphi(),cj->rank());
        l1GctAllJetsOccEtaPhi_->Fill(cj->regionId().ieta(),cj->regionId().iphi());
        l1GctCenJetsEtEtaPhi_->Fill(cj->regionId().ieta(),cj->regionId().iphi(),cj->rank());
        l1GctCenJetsOccEtaPhi_->Fill(cj->regionId().ieta(),cj->regionId().iphi());
      }
    }
    if ( l1CenJets->size()==4){
      // Rank for each candidate
      l1GctCenJetsRankCand0_->Fill((*l1CenJets).at(0).rank());
      l1GctCenJetsRankCand1_->Fill((*l1CenJets).at(1).rank());
      l1GctCenJetsRankCand2_->Fill((*l1CenJets).at(2).rank());
      l1GctCenJetsRankCand3_->Fill((*l1CenJets).at(3).rank());
    }
  } else {    
    edm::LogWarning("DataNotFound") << " Could not find l1CenJets label was " << gctCenJetsSource_ ;
  }

  // Forward jets
  if (l1ForJets.isValid()) {
    for (L1GctJetCandCollection::const_iterator fj = l1ForJets->begin(); fj != l1ForJets->end(); fj++) {
      l1GctForJetsRank_->Fill(fj->rank());
      // only plot eta and phi maps for non-zero candidates
      if (fj->rank()) {
        l1GctAllJetsEtEtaPhi_->Fill(fj->regionId().ieta(),fj->regionId().iphi(),fj->rank());
        l1GctAllJetsOccEtaPhi_->Fill(fj->regionId().ieta(),fj->regionId().iphi());
        l1GctForJetsEtEtaPhi_->Fill(fj->regionId().ieta(),fj->regionId().iphi(),fj->rank());
        l1GctForJetsOccEtaPhi_->Fill(fj->regionId().ieta(),fj->regionId().iphi());
      }
    }
    if ( l1ForJets->size()==4){
      // Rank for each candidate
      l1GctForJetsRankCand0_->Fill((*l1ForJets).at(0).rank());
      l1GctForJetsRankCand1_->Fill((*l1ForJets).at(1).rank());
      l1GctForJetsRankCand2_->Fill((*l1ForJets).at(2).rank());
      l1GctForJetsRankCand3_->Fill((*l1ForJets).at(3).rank());
    }
  } else {    
    edm::LogWarning("DataNotFound") << " Could not find l1ForJets label was " << gctForJetsSource_ ;
  }

  // Tau jets
  if (l1TauJets.isValid()) {
    for (L1GctJetCandCollection::const_iterator tj = l1TauJets->begin(); tj != l1TauJets->end(); tj++) {
      l1GctTauJetsRank_->Fill(tj->rank());
      // only plot eta and phi maps for non-zero candidates
      if (tj->rank()) {
        l1GctTauJetsEtEtaPhi_->Fill(tj->regionId().ieta(),tj->regionId().iphi(),tj->rank());
        l1GctTauJetsOccEtaPhi_->Fill(tj->regionId().ieta(),tj->regionId().iphi());
      }
    }
    if (l1TauJets->size()==4){
      // Rank for each candidate
      l1GctTauJetsRankCand0_->Fill((*l1TauJets).at(0).rank());
      l1GctTauJetsRankCand1_->Fill((*l1TauJets).at(1).rank());
      l1GctTauJetsRankCand2_->Fill((*l1TauJets).at(2).rank());
      l1GctTauJetsRankCand3_->Fill((*l1TauJets).at(3).rank());
    }
  } else {    
    edm::LogWarning("DataNotFound") << " Could not find l1TauJets label was " << gctTauJetsSource_ ;
  }
  
  // Missing ET
  if (l1EtMiss.isValid()) { 
    if (l1EtMiss->size()) {
      if (l1EtMiss->at(0).overFlow() == 0 && l1EtMiss->at(0).et() > 0) {
        //Avoid problems with met=0 candidates affecting MET_PHI plots
        l1GctEtMiss_->Fill(l1EtMiss->at(0).et());
        l1GctEtMissPhi_->Fill(l1EtMiss->at(0).phi());
      }
      l1GctEtMissOf_->Fill(l1EtMiss->at(0).overFlow());
    }
  } else {
    edm::LogWarning("DataNotFound") << " Could not find l1EtMiss label was " << gctEnergySumsSource_ ;    
  }

  // Missing HT
  if (l1HtMiss.isValid()) { 
    if (l1HtMiss->size()) {
      if (l1HtMiss->at(0).overFlow() == 0 && l1HtMiss->at(0).et() > 0) {
        //Avoid problems with mht=0 candidates affecting MHT_PHI plots
         l1GctHtMiss_->Fill(l1HtMiss->at(0).et());
         l1GctHtMissPhi_->Fill(l1HtMiss->at(0).phi());
      }
      l1GctHtMissOf_->Fill(l1HtMiss->at(0).overFlow());
    }
  } else {
    edm::LogWarning("DataNotFound") << " Could not find l1HtMiss label was " << gctEnergySumsSource_ ;    
  }

  // Missing ET HT correlations
  if (l1HtMiss.isValid() && l1EtMiss.isValid()) { 
    if (l1HtMiss->size() && l1HtMiss->size()) {
      if (l1HtMiss->at(0).overFlow() == 0 && l1EtMiss->at(0).overFlow() == 0) {
        // Avoid problems overflows
         l1GctEtMissHtMissCorr_->Fill(l1EtMiss->at(0).et(),l1HtMiss->at(0).et());
         l1GctEtMissHtMissCorrPhi_->Fill(l1EtMiss->at(0).phi(),l1HtMiss->at(0).phi());
      }
    }
  } else {
    edm::LogWarning("DataNotFound") << " Could not find l1EtMiss or l1HtMiss label was " << gctEnergySumsSource_ ;    
  }

  // HT 
  if (l1EtHad.isValid()) {
    if (l1EtHad->size()) { 
      if (l1EtHad->at(0).overFlow() == 0) l1GctEtHad_->Fill(l1EtHad->at(0).et());
      l1GctEtHadOf_->Fill(l1EtHad->at(0).overFlow());
    }
  } else {
    edm::LogWarning("DataNotFound") << " Could not find l1EtHad label was " << gctEnergySumsSource_ ;    
  }

  // Total ET
  if (l1EtTotal.isValid()) {
    if (l1EtTotal->size()) { 
      if (l1EtTotal->at(0).overFlow() == 0) l1GctEtTotal_->Fill(l1EtTotal->at(0).et());
      l1GctEtTotalOf_->Fill(l1EtTotal->at(0).overFlow());
    }
  } else {
    edm::LogWarning("DataNotFound") << " Could not find l1EtTotal label was " << gctEnergySumsSource_ ;    
  }

  //HF Ring Et Sums
  if (l1HFSums.isValid()) {
    for (L1GctHFRingEtSumsCollection::const_iterator hfs=l1HFSums->begin(); hfs!=l1HFSums->end(); hfs++){ 
      // Individual ring Et sums
      l1GctHFRing1ETSumPosEta_->Fill(hfs->etSum(0));
      l1GctHFRing1ETSumNegEta_->Fill(hfs->etSum(1));
      l1GctHFRing2ETSumPosEta_->Fill(hfs->etSum(2));
      l1GctHFRing2ETSumNegEta_->Fill(hfs->etSum(3));
      // Ratio of ring Et sums
      if (hfs->etSum(2)!=0) l1GctHFRingRatioPosEta_->Fill((hfs->etSum(0))/(hfs->etSum(2)));
      if (hfs->etSum(3)!=0) l1GctHFRingRatioNegEta_->Fill((hfs->etSum(1))/(hfs->etSum(3)));
      // Correlate positive and neagative eta
      l1GctHFRing1PosEtaNegEta_->Fill(hfs->etSum(0),hfs->etSum(1));
      l1GctHFRing2PosEtaNegEta_->Fill(hfs->etSum(2),hfs->etSum(3));
    }
  } else {    
    edm::LogWarning("DataNotFound") << " Could not find l1HFSums label was " << gctEnergySumsSource_ ;
  }

  // HF Ring Counts
  if (l1HFCounts.isValid()) {
    for (L1GctHFBitCountsCollection::const_iterator hfc=l1HFCounts->begin(); hfc!=l1HFCounts->end(); hfc++){ 
      // Individual ring counts
      l1GctHFRing1TowerCountPosEta_->Fill(hfc->bitCount(0));
      l1GctHFRing1TowerCountNegEta_->Fill(hfc->bitCount(1));
      l1GctHFRing2TowerCountPosEta_->Fill(hfc->bitCount(2));
      l1GctHFRing2TowerCountNegEta_->Fill(hfc->bitCount(3));
      // Correlate positive and negative eta
      l1GctHFRing1TowerCountPosEtaNegEta_->Fill(hfc->bitCount(0),hfc->bitCount(1));
      l1GctHFRing2TowerCountPosEtaNegEta_->Fill(hfc->bitCount(2),hfc->bitCount(3));
    }
  } else {    
    edm::LogWarning("DataNotFound") << " Could not find l1HFCounts label was " << gctEnergySumsSource_ ;
  }

  // Isolated EM
  if (l1IsoEm.isValid()) {
    for (L1GctEmCandCollection::const_iterator ie=l1IsoEm->begin(); ie!=l1IsoEm->end(); ie++) {
      l1GctIsoEmRank_->Fill(ie->rank());
      // only plot eta and phi maps for non-zero candidates
      if (ie->rank()){ 
        l1GctIsoEmRankEtaPhi_->Fill(ie->regionId().ieta(),ie->regionId().iphi(),ie->rank());
        l1GctIsoEmOccEtaPhi_->Fill(ie->regionId().ieta(),ie->regionId().iphi());
      }
    }
    if (l1IsoEm->size()==4){
      // Rank for each candidate
      l1GctIsoEmRankCand0_->Fill((*l1IsoEm).at(0).rank());
      l1GctIsoEmRankCand1_->Fill((*l1IsoEm).at(1).rank());
      l1GctIsoEmRankCand2_->Fill((*l1IsoEm).at(2).rank());
      l1GctIsoEmRankCand3_->Fill((*l1IsoEm).at(3).rank());
    }
  } else {
    edm::LogWarning("DataNotFound") << " Could not find l1IsoEm label was " << gctIsoEmSource_ ;
  } 

  // Non-isolated EM
  if (l1NonIsoEm.isValid()) { 
    for (L1GctEmCandCollection::const_iterator ne=l1NonIsoEm->begin(); ne!=l1NonIsoEm->end(); ne++) {
      l1GctNonIsoEmRank_->Fill(ne->rank());
      // only plot eta and phi maps for non-zero candidates
      if (ne->rank()){ 
        l1GctNonIsoEmRankEtaPhi_->Fill(ne->regionId().ieta(),ne->regionId().iphi(),ne->rank());
        l1GctNonIsoEmOccEtaPhi_->Fill(ne->regionId().ieta(),ne->regionId().iphi());
      }
    }
    if (l1NonIsoEm->size()==4){
      // Rank for each candidate
      l1GctNonIsoEmRankCand0_->Fill((*l1NonIsoEm).at(0).rank());
      l1GctNonIsoEmRankCand1_->Fill((*l1NonIsoEm).at(1).rank());
      l1GctNonIsoEmRankCand2_->Fill((*l1NonIsoEm).at(2).rank());
      l1GctNonIsoEmRankCand3_->Fill((*l1NonIsoEm).at(3).rank());
    }
  } else {
    edm::LogWarning("DataNotFound") << " Could not find l1NonIsoEm label was " << gctNonIsoEmSource_ ;
  }     
}

  
