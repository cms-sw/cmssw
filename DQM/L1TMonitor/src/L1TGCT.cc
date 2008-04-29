/*
 * \file L1TGCT.cc
 *
 * $Date: 2008/04/28 09:23:07 $
 * $Revision: 1.24 $
 * \author J. Berryhill
 *
 * $Log: L1TGCT.cc,v $
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

// Ranks 6, 10 and 12 bits
const unsigned int R6BINS = 64;
const float R6MIN = -0.5;
const float R6MAX = 63.5;
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
    std::cout << "L1TGCT: constructor...." << std::endl;


  dbe = NULL;
  if (ps.getUntrackedParameter < bool > ("DQMStore", false)) {
    dbe = edm::Service < DQMStore > ().operator->();
    dbe->setVerbose(0);
  }

  outputFile_ = ps.getUntrackedParameter < std::string > ("outputFile", "");
  if (outputFile_.size() != 0) {
    std::cout << "L1T Monitoring histograms will be saved to "
	      << outputFile_ << std::endl;
  }

  bool disable =
    ps.getUntrackedParameter < bool > ("disableROOToutput", false);
  if (disable) {
    outputFile_ = "";
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

    dbe->setCurrentFolder("L1T/EventInfo/errorSummarySegments");

    l1GctSummIsoEmRankEtaPhi_ = dbe->book2D("IsoEmRankEtaPhiSumm", "ISO EM RANK", 
                                            PHIBINS, PHIMIN, PHIMAX, 		    
                                            ETABINS, ETAMIN, ETAMAX);
    l1GctSummNonIsoEmRankEtaPhi_ = dbe->book2D("NonIsoEmRankEtaPhiSumm", "NON-ISO EM RANK",
                                               PHIBINS, PHIMIN, PHIMAX, 
                                               ETABINS, ETAMIN, ETAMAX);    

    dbe->setCurrentFolder("L1T/L1TGCT");

    // GCT hardware quantities for experts
    l1GctCenJetsEtEtaPhi_ = dbe->book2D("CenJetsEtEtaPhi", "CENTRAL JET RANK",
					PHIBINS, PHIMIN, PHIMAX, 
					ETABINS, ETAMIN, ETAMAX);
    l1GctForJetsEtEtaPhi_ = dbe->book2D("ForJetsEtEtaPhi", "FORWARD JET RANK",
					PHIBINS, PHIMIN, PHIMAX, 
					ETABINS, ETAMIN, ETAMAX);
    l1GctTauJetsEtEtaPhi_ = dbe->book2D("TauJetsEtEtaPhi", "TAU JET RANK", 
					PHIBINS, PHIMIN, PHIMAX, 
					ETABINS, ETAMIN, ETAMAX);
    l1GctIsoEmRankEtaPhi_ = dbe->book2D("IsoEmRankEtaPhi", "ISO EM RANK", 
                                        PHIBINS, PHIMIN, PHIMAX, 		    
                                        ETABINS, ETAMIN, ETAMAX);
    l1GctNonIsoEmRankEtaPhi_ = dbe->book2D("NonIsoEmRankEtaPhi", "NON-ISO EM RANK",
                                           PHIBINS, PHIMIN, PHIMAX, 
                                           ETABINS, ETAMIN, ETAMAX);
    l1GctCenJetsOccEtaPhi_ = dbe->book2D("CenJetsOccEtaPhi", "CENTRAL JET OCCUPANCY",
					 PHIBINS, PHIMIN, PHIMAX, 
					 ETABINS, ETAMIN, ETAMAX);
    l1GctForJetsOccEtaPhi_ = dbe->book2D("ForJetsOccEtaPhi", "FORWARD JET OCCUPANCY",
					 PHIBINS, PHIMIN, PHIMAX,
					 ETABINS, ETAMIN, ETAMAX);
    l1GctTauJetsOccEtaPhi_ = dbe->book2D("TauJetsOccEtaPhi", "TAU JET OCCUPANCY",
					 PHIBINS, PHIMIN, PHIMAX, 
					 ETABINS, ETAMIN, ETAMAX);
    l1GctIsoEmOccEtaPhi_ = dbe->book2D("IsoEmOccEtaPhi", "ISO EM OCCUPANCY",
				       PHIBINS, PHIMIN, PHIMAX, 
				       ETABINS, ETAMIN, ETAMAX);
    l1GctNonIsoEmOccEtaPhi_ = dbe->book2D("NonIsoEmOccEtaPhi", "NON-ISO EM OCCUPANCY",
					  PHIBINS, PHIMIN, PHIMAX, 
					  ETABINS, ETAMIN, ETAMAX);

    // For Qtests need 1D eta and phi histograms (would be better if Qtests ran on 2D histograms too!)
    l1GctCenJetsOccEta_  = dbe->book1D("CenJetsOccEta", "CENTRAL JET ETA OCCUPANCY", ETABINS, ETAMIN, ETAMAX);
    l1GctCenJetsOccPhi_  = dbe->book1D("CenJetsOccPhi", "CENTRAL JET PHI OCCUPANCY", PHIBINS, PHIMIN, PHIMAX); 
    l1GctForJetsOccEta_  = dbe->book1D("ForJetsOccEta", "FORWARD JET ETA OCCUPANCY", ETABINS, ETAMIN, ETAMAX);
    l1GctForJetsOccPhi_  = dbe->book1D("ForJetsOccPhi", "FORWARD JET PHI OCCUPANCY", PHIBINS, PHIMIN, PHIMAX);
    l1GctTauJetsOccEta_  = dbe->book1D("TauJetsOccEta", "TAU JET ETA OCCUPANCY", ETABINS, ETAMIN, ETAMAX);
    l1GctTauJetsOccPhi_  = dbe->book1D("TauJetsOccPhi", "TAU JET PHI OCCUPANCY", PHIBINS, PHIMIN, PHIMAX); 
    l1GctIsoEmOccEta_    = dbe->book1D("IsoEmOccEta", "ISO EM ETA OCCUPANCY", ETABINS, ETAMIN, ETAMAX);
    l1GctIsoEmOccPhi_    = dbe->book1D("IsoEmOccPhi", "ISO EM PHI OCCUPANCY", PHIBINS, PHIMIN, PHIMAX); 
    l1GctNonIsoEmOccEta_ = dbe->book1D("NonIsoEmOccEta", "NON-ISO EM ETA OCCUPANCY", ETABINS, ETAMIN, ETAMAX);
    l1GctNonIsoEmOccPhi_ = dbe->book1D("NonIsoEmOccPhi", "NON-ISO EM PHI OCCUPANCY", PHIBINS, PHIMIN, PHIMAX); 

    // Rank histograms
    l1GctCenJetsRank_  = dbe->book1D("CenJetsRank", "CENTRAL JET RANK", R6BINS, R6MIN, R6MAX);
    l1GctForJetsRank_  = dbe->book1D("ForJetsRank", "FORWARD JET RANK", R6BINS, R6MIN, R6MAX);
    l1GctTauJetsRank_  = dbe->book1D("TauJetsRank", "TAU JET RANK", R6BINS, R6MIN, R6MAX);
    l1GctIsoEmRank_    = dbe->book1D("IsoEmRank", "ISO EM RANK", R6BINS, R6MIN, R6MAX);
    l1GctNonIsoEmRank_ = dbe->book1D("NonIsoEmRank", "NON-ISO EM RANK", R6BINS, R6MIN, R6MAX);

    // Energy sums
    l1GctEtMiss_    = dbe->book1D("EtMiss", "MISSING E_{T}", R12BINS, R12MIN, R12MAX);
    l1GctEtMissPhi_ = dbe->book1D("EtMissPhi", "MISSING E_{T} #phi", METPHIBINS, PHIMIN, PHIMAX);
    l1GctEtTotal_   = dbe->book1D("EtTotal", "TOTAL E_{T}", R12BINS, R12MIN, R12MAX);
    l1GctEtHad_     = dbe->book1D("EtHad", "TOTAL HAD E_{T}", R12BINS, R12MIN, R12MAX);

    // More detailed EM quantities
    l1GctIsoEmRankCand0_ = dbe->book1D("GctIsoEmRankCand0","ISO EM RANK CAND 0", R6BINS, R6MIN, R6MAX);
    l1GctIsoEmRankCand1_ = dbe->book1D("GctIsoEmRankCand1","ISO EM RANK CAND 1", R6BINS, R6MIN, R6MAX);
    l1GctIsoEmRankCand2_ = dbe->book1D("GctIsoEmRankCand2","ISO EM RANK CAND 2", R6BINS, R6MIN, R6MAX);
    l1GctIsoEmRankCand3_ = dbe->book1D("GctIsoEmRankCand3","ISO EM RANK CAND 3", R6BINS, R6MIN, R6MAX);

    l1GctNonIsoEmRankCand0_ = dbe->book1D("GctNonIsoEmRankCand0","NON-ISO EM RANK CAND 0", R6BINS, R6MIN, R6MAX);
    l1GctNonIsoEmRankCand1_ = dbe->book1D("GctNonIsoEmRankCand1","NON-ISO EM RANK CAND 1", R6BINS, R6MIN, R6MAX);
    l1GctNonIsoEmRankCand2_ = dbe->book1D("GctNonIsoEmRankCand2","NON-ISO EM RANK CAND 2", R6BINS, R6MIN, R6MAX);
    l1GctNonIsoEmRankCand3_ = dbe->book1D("GctNonIsoEmRankCand3","NON-ISO EM RANK CAND 3", R6BINS, R6MIN, R6MAX);

    l1GctIsoEmRankDiff01_ = dbe->book1D("GctIsoEmRankDiffCand01","ISO EM RANK CAND 0 - CAND 1", 2*R6BINS, -R6MAX, R6MAX);
    l1GctIsoEmRankDiff12_ = dbe->book1D("GctIsoEmRankDiffCand12","ISO EM RANK CAND 1 - CAND 2", 2*R6BINS, -R6MAX, R6MAX);
    l1GctIsoEmRankDiff23_ = dbe->book1D("GctIsoEmRankDiffCand23","ISO EM RANK CAND 2 - CAND 3", 2*R6BINS, -R6MAX, R6MAX);

    l1GctNonIsoEmRankDiff01_ = dbe->book1D("GctNonIsoEmRankDiffCand01","NON-ISO EM RANK CAND 0 - CAND 1", 2*R6BINS, -R6MAX, R6MAX);
    l1GctNonIsoEmRankDiff12_ = dbe->book1D("GctNonIsoEmRankDiffCand12","NON-ISO EM RANK CAND 1 - CAND 2", 2*R6BINS, -R6MAX, R6MAX);
    l1GctNonIsoEmRankDiff23_ = dbe->book1D("GctNonIsoEmRankDiffCand23","NON-ISO EM RANK CAND 2 - CAND 3", 2*R6BINS, -R6MAX, R6MAX);    

    dbe->setCurrentFolder("L1T/L1TGCT/ISO EM");

    for (unsigned int eta=0; eta<ETABINS; eta++){
      for (unsigned int phi=0; phi<PHIBINS; phi++){
        std::stringstream hName; hName << "GctIsoEmRank" << "-" << eta << "-" << phi;
        std::stringstream hTitle; hTitle << "ISO EM RANK " << "Eta=" << eta << " Phi=" << phi;
        l1GctIsoEmRankBin_[eta][phi] = dbe->book1D(hName.str(),hTitle.str(),R6BINS,R6MIN,R6MAX);
      }
    }

    dbe->setCurrentFolder("L1T/L1TGCT/NON-ISO EM");

    for (unsigned int eta=0; eta<ETABINS; eta++){
      for (unsigned int phi=0; phi<PHIBINS; phi++){
        std::stringstream hName; hName << "GctNonIsoEmRank" << "-" << eta << "-" << phi;
        std::stringstream hTitle; hTitle << "NON-ISO EM RANK " << "Eta=" << eta << " Phi=" << phi;
        l1GctNonIsoEmRankBin_[eta][phi] = dbe->book1D(hName.str(),hTitle.str(),R6BINS,R6MIN,R6MAX);
      }
    }

  }

}


void L1TGCT::endJob(void)
{
  if (verbose_)
    std::cout << "L1TGCT: end job...." << std::endl;
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
    std::cout << "L1TGCT: analyze...." << std::endl;
  }
  
  // update to those generated in GctRawToDigi
  edm::Handle < L1GctEmCandCollection > l1IsoEm;
  edm::Handle < L1GctEmCandCollection > l1NonIsoEm;
  edm::Handle < L1GctJetCandCollection > l1CenJets;
  edm::Handle < L1GctJetCandCollection > l1ForJets;
  edm::Handle < L1GctJetCandCollection > l1TauJets;
  edm::Handle < L1GctEtMiss >  l1EtMiss;
  edm::Handle < L1GctEtHad >   l1EtHad;
  edm::Handle < L1GctEtTotal > l1EtTotal;

  // Split this into two parts as this appears to be the way the HW works.
  // This should not be necessary. The unpacker should produce all 
  // collections regardless of input data by default but I leave this for now
  bool doJet = true;
  bool doEm = true;
  
  e.getByLabel(gctCenJetsSource_, l1CenJets);
  e.getByLabel(gctForJetsSource_, l1ForJets);
  e.getByLabel(gctTauJetsSource_, l1TauJets);
  
  e.getByLabel(gctEnergySumsSource_, l1EtMiss);
  e.getByLabel(gctEnergySumsSource_, l1EtHad);
  e.getByLabel(gctEnergySumsSource_, l1EtTotal);
   
  if (!l1CenJets.isValid())  {
    edm::LogInfo("DataNotFound") << " Could not find l1CenJets"
      ", label was " << gctCenJetsSource_ ;
    doJet = false;
  }
   
  if (!l1ForJets.isValid())  {
    edm::LogInfo("DataNotFound") << " Could not find l1ForJets"
      ", label was " << gctForJetsSource_ ;
    doJet = false;
  }
   
  if (!l1TauJets.isValid())  {
    edm::LogInfo("DataNotFound") << " Could not find l1TauJets"
      ", label was " << gctTauJetsSource_ ;
    doJet = false;
  }
   
  if (!l1EtMiss.isValid())  {
    edm::LogInfo("DataNotFound") << " Could not find l1EtMiss"
      ", label was " << gctEnergySumsSource_ ;
    doJet = false;
  }
     
  if (!l1EtHad.isValid())  {
    edm::LogInfo("DataNotFound") << " Could not find l1EtHad"
      ", label was " << gctEnergySumsSource_ ;
    doJet = false;
  }
   
  if (!l1EtTotal.isValid())  {
    edm::LogInfo("DataNotFound") << " Could not find l1EtTotal"
      ", label was " << gctEnergySumsSource_ ;
    doJet = false;
  }

  // EM data
  
  e.getByLabel(gctIsoEmSource_, l1IsoEm);
  e.getByLabel(gctNonIsoEmSource_, l1NonIsoEm);
  
  if (!l1IsoEm.isValid()) {
    edm::LogInfo("DataNotFound") << " Could not find l1IsoEm "
      " elements, label was " << gctIsoEmSource_ ;
    doEm = false;
  }
  if (!l1NonIsoEm.isValid()) {
    edm::LogInfo("DataNotFound") << " Could not find l1NonIsoEm "
      " elements, label was " << gctNonIsoEmSource_ ;
    doEm = false;
  }

  if ( (! doEm) && (! doJet) ) {
    if (  verbose_ )
      std::cout << "L1TGCT: Bailing, didn't find squat."<<std::endl;
    return;
  }
    
  // Fill the histograms for the jets
  if ( doJet ) {
    // Central jets
    if ( verbose_ ) {
      std::cout << "L1TGCT: number of central jets = " 
		<< l1CenJets->size() << std::endl;
    }
    for (L1GctJetCandCollection::const_iterator cj = l1CenJets->begin();
	 cj != l1CenJets->end(); cj++) {
      if ( cj->rank() == 0 ) continue;
      l1GctCenJetsEtEtaPhi_->Fill(cj->regionId().iphi(),cj->regionId().ieta(),cj->rank());
      l1GctCenJetsOccEtaPhi_->Fill(cj->regionId().iphi(),cj->regionId().ieta());
      l1GctCenJetsOccEta_->Fill(cj->regionId().ieta());
      l1GctCenJetsOccPhi_->Fill(cj->regionId().iphi());
      l1GctCenJetsRank_->Fill(cj->rank());
      if ( verbose_ ) {
	std::cout << "L1TGCT: Central jet " 
		  << cj->regionId().iphi() << ", " << cj->regionId().ieta()
		  << ", " << cj->rank() << std::endl;
      }
    }

    // Forward jets
    if ( verbose_ ) {
      std::cout << "L1TGCT: number of forward jets = " 
		<< l1ForJets->size() << std::endl;
    }
    for (L1GctJetCandCollection::const_iterator fj = l1ForJets->begin();
	 fj != l1ForJets->end(); fj++) {
      if ( fj->rank() == 0 ) continue;
      l1GctForJetsEtEtaPhi_->Fill(fj->regionId().iphi(),fj->regionId().ieta(),fj->rank());
      l1GctForJetsOccEtaPhi_->Fill(fj->regionId().iphi(),fj->regionId().ieta());
      l1GctForJetsOccEta_->Fill(fj->regionId().ieta());
      l1GctForJetsOccPhi_->Fill(fj->regionId().iphi());
      l1GctForJetsRank_->Fill(fj->rank());
      if ( verbose_ ) {
	std::cout << "L1TGCT: Forward jet " 
		  << fj->regionId().iphi() << ", " << fj->regionId().ieta()
		  << ", " << fj->rank() << std::endl;
      }
    }

    // Tau jets
    if ( verbose_ ) {
      std::cout << "L1TGCT: number of tau jets = " 
		<< l1TauJets->size() << std::endl;
    }
    for (L1GctJetCandCollection::const_iterator tj = l1TauJets->begin();
	 tj != l1TauJets->end(); tj++) {
      if ( tj->rank() == 0 ) continue;
      l1GctTauJetsEtEtaPhi_->Fill(tj->regionId().iphi(),tj->regionId().ieta(),tj->rank());
      l1GctTauJetsOccEtaPhi_->Fill(tj->regionId().iphi(),tj->regionId().ieta());
      l1GctTauJetsOccEta_->Fill(tj->regionId().ieta());
      l1GctTauJetsOccPhi_->Fill(tj->regionId().iphi());
      l1GctTauJetsRank_->Fill(tj->rank());
      if ( verbose_ ) {
	std::cout << "L1TGCT: Tau jet " 
		  << tj->regionId().iphi() << ", " << tj->regionId().ieta()
		  << ", " << tj->rank() << std::endl;
      }
    }

    // Energy sums
    l1GctEtMiss_->Fill(l1EtMiss->et());
    l1GctEtMissPhi_->Fill(l1EtMiss->phi());

    // these don't have phi values
    l1GctEtHad_->Fill(l1EtHad->et());
    l1GctEtTotal_->Fill(l1EtTotal->et());

  }


  if ( doEm ) {
    // Isolated EM
    if ( verbose_ ) {
      std::cout << "L1TGCT: number of iso em cands: " 
		<< l1IsoEm->size() << std::endl;
    }
    for (L1GctEmCandCollection::const_iterator ie=l1IsoEm->begin(); ie!=l1IsoEm->end(); ie++) {
      l1GctSummIsoEmRankEtaPhi_->Fill(ie->regionId().iphi(),ie->regionId().ieta(),ie->rank());
      l1GctIsoEmRankEtaPhi_->Fill(ie->regionId().iphi(),ie->regionId().ieta(),ie->rank());
      l1GctIsoEmOccEtaPhi_->Fill(ie->regionId().iphi(),ie->regionId().ieta());
      l1GctIsoEmOccEta_->Fill(ie->regionId().ieta());
      l1GctIsoEmOccPhi_->Fill(ie->regionId().iphi());
      l1GctIsoEmRank_->Fill(ie->rank());
      l1GctIsoEmRankBin_[ie->regionId().ieta()][ie->regionId().iphi()]->Fill(ie->rank());
      if ( verbose_ ) {
	std::cout << "L1TGCT: iso em " 
		  << ie->regionId().iphi() << ", " 
		  << ie->regionId().ieta() << ", " << ie->rank()
		  << std::endl;
      }

    } 

    // Rank for each candidate
    l1GctIsoEmRankCand0_->Fill((*l1IsoEm)[0].rank());
    l1GctIsoEmRankCand1_->Fill((*l1IsoEm)[1].rank());
    l1GctIsoEmRankCand2_->Fill((*l1IsoEm)[2].rank());
    l1GctIsoEmRankCand3_->Fill((*l1IsoEm)[3].rank());

    // Differences between candidate ranks
    l1GctIsoEmRankDiff01_->Fill((*l1IsoEm)[0].rank()-(*l1IsoEm)[1].rank());
    l1GctIsoEmRankDiff12_->Fill((*l1IsoEm)[1].rank()-(*l1IsoEm)[2].rank());
    l1GctIsoEmRankDiff23_->Fill((*l1IsoEm)[2].rank()-(*l1IsoEm)[3].rank());

    // Non-isolated EM
    if ( verbose_ ) {
      std::cout << "L1TGCT: number of non-iso em cands: " 
		<< l1NonIsoEm->size() << std::endl;
    }
    for (L1GctEmCandCollection::const_iterator ne=l1NonIsoEm->begin(); ne!=l1NonIsoEm->end(); ne++) {
      l1GctSummNonIsoEmRankEtaPhi_->Fill(ne->regionId().iphi(),ne->regionId().ieta(),ne->rank());
      l1GctNonIsoEmRankEtaPhi_->Fill(ne->regionId().iphi(),ne->regionId().ieta(),ne->rank());
      l1GctNonIsoEmOccEtaPhi_->Fill(ne->regionId().iphi(),ne->regionId().ieta());
      l1GctNonIsoEmOccEta_->Fill(ne->regionId().ieta());
      l1GctNonIsoEmOccPhi_->Fill(ne->regionId().iphi());
      l1GctNonIsoEmRank_->Fill(ne->rank());
      l1GctNonIsoEmRankBin_[ne->regionId().ieta()][ne->regionId().iphi()]->Fill(ne->rank());

      if ( verbose_ ) {
	std::cout << "L1TGCT: non-iso em " 
		  << ne->regionId().iphi() << ", " 
		  << ne->regionId().ieta() << ", " << ne->rank()
		  << std::endl;
      }
    } 

    // Rank for each candidate
    l1GctNonIsoEmRankCand0_->Fill((*l1NonIsoEm)[0].rank());
    l1GctNonIsoEmRankCand1_->Fill((*l1NonIsoEm)[1].rank());
    l1GctNonIsoEmRankCand2_->Fill((*l1NonIsoEm)[2].rank());
    l1GctNonIsoEmRankCand3_->Fill((*l1NonIsoEm)[3].rank());
   
    // Differences between candidate ranks
    l1GctNonIsoEmRankDiff01_->Fill((*l1NonIsoEm)[0].rank()-(*l1NonIsoEm)[1].rank());
    l1GctNonIsoEmRankDiff12_->Fill((*l1NonIsoEm)[1].rank()-(*l1NonIsoEm)[2].rank());
    l1GctNonIsoEmRankDiff23_->Fill((*l1NonIsoEm)[2].rank()-(*l1NonIsoEm)[3].rank());

  }
}

  
