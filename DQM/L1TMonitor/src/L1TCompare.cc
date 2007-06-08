/*
 * \file L1TCompare.cc
 * $Id: L1TCompare.cc,v 1.1 2007/06/06 14:55:51 wittich Exp $
 * \author P. Wittich
 * \brief Compare different parts of the trigger chain (e.g., RCT-GCT )
 * $Log: L1TCompare.cc,v $
 * Revision 1.1  2007/06/06 14:55:51  wittich
 * compare within trigger subsystems
 *
 */

#include "DQM/L1TMonitor/interface/L1TCompare.h"

// GCT and RCT data formats
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtSums.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloRegionDetId.h"

// L1Extra
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"

// Ecal
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

// stl
#include <algorithm>

using namespace l1extra;
using namespace edm;

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

// For GCT this should be 15 bins, -14.5 to 14.5
//
const unsigned int ETABINS = 22;
const float ETAMIN = -0.5;
const float ETAMAX = 21.5;

// TPG
const unsigned int TPPHIBINS = 72;
const float TPPHIMIN = 0.5;
const float TPPHIMAX = 72.5;

const unsigned int TPETABINS = 65;
const float TPETAMIN = -32.5;
const float TPETAMAX = 32.5;


L1TCompare::L1TCompare(const ParameterSet & ps) :
  rctSource_( ps.getParameter< InputTag >("rctSource") )
  ,gctSource_( ps.getParameter< InputTag >("gctSource") )
  ,ecalTpgSource_(ps.getParameter<edm::InputTag>("ecalTpgSource"))

{

  // verbosity switch
  verbose_ = ps.getUntrackedParameter < bool > ("verbose", false);

  if (verbose())
    std::cout << "L1TCompare: constructor...." << std::endl;

  logFile_.open("L1TCompare.log");

  dbe = NULL;
  if (ps.getUntrackedParameter < bool > ("DaqMonitorBEInterface", false)) {
    dbe = Service < DaqMonitorBEInterface > ().operator->();
    dbe->setVerbose(0);
  }

  monitorDaemon_ = false;
  if (ps.getUntrackedParameter < bool > ("MonitorDaemon", false)) {
    Service < MonitorDaemon > daemon;
    daemon.operator->();
    monitorDaemon_ = true;
  }

  outputFile_ =
      ps.getUntrackedParameter < std::string > ("outputFile", "");
  if (outputFile_.size() != 0) {
    std::
	cout << "L1T Monitoring histograms will be saved to " <<
	outputFile_.c_str() << std::endl;
  }
  else {
    outputFile_ = "L1TDQM.root";
  }

  bool disable =
      ps.getUntrackedParameter < bool > ("disableROOToutput", false);
  if (disable) {
    outputFile_ = "";
  }


  if (dbe != NULL) {
    dbe->setCurrentFolder("L1TMonitor/L1TRCT");
  }


}

L1TCompare::~L1TCompare()
{
}

void L1TCompare::beginJob(const EventSetup & c)
{

  nev_ = 0;

  // get hold of back-end interface
  DaqMonitorBEInterface *dbe = 0;
  dbe = Service < DaqMonitorBEInterface > ().operator->();

  if (dbe) {
    dbe->setCurrentFolder("L1TMonitor/Compare");
    dbe->rmdir("L1TMonitor/Compare");
  }


  if (dbe) {
    dbe->setCurrentFolder("L1TMonitor/Compare");
    
    // -------------------------------------------
    // RCT-GCT
    // -------------------------------------------
    // Isolated
    rctGctLeadingIsoEmRank_ = dbe->book2D("rctGctLeadingIsoEmRank",
				       "RCT-GCT: rank", R6BINS, R6MIN, R6MAX,
				       R6BINS, R6MIN, R6MAX);
    rctGctLeadingIsoEmEta_ = dbe->book2D("rctGctLeadingIsoEmEta",
				      "RCT-GCT: #eta", ETABINS, ETAMIN, ETAMAX,
				       ETABINS, ETAMIN, ETAMAX);
    rctGctLeadingIsoEmPhi_ = dbe->book2D("rctGctLeadingIsoEmPhi",
				      "RCT-GCT: #phi", PHIBINS, PHIMIN, PHIMAX,
				       PHIBINS, PHIMIN, PHIMAX);
    // non-Isolated
    rctGctLeadingNonIsoEmRank_ = dbe->book2D("rctGctLeadingNonIsoEmRank",
				       "RCT-GCT: rank", R6BINS, R6MIN, R6MAX,
				       R6BINS, R6MIN, R6MAX);
    rctGctLeadingNonIsoEmEta_ = dbe->book2D("rctGctLeadingNonIsoEmEta",
				      "RCT-GCT: #eta", ETABINS, ETAMIN, ETAMAX,
				       ETABINS, ETAMIN, ETAMAX);
    rctGctLeadingNonIsoEmPhi_ = dbe->book2D("rctGctLeadingNonIsoEmPhi",
				      "RCT-GCT: #phi", PHIBINS, PHIMIN, PHIMAX,
				       PHIBINS, PHIMIN, PHIMAX);
    // -------------------------------------------
    // ECAL TPG - RCT
    // -------------------------------------------
    ecalTpgRctLeadingEmRank_ = dbe->book2D("ecalTpgRctLeadingEmRank",
					   "ECAL TPG-RCT: rank", 
					   R6BINS, R6MIN, R6MAX,
					   R6BINS, R6MIN, R6MAX);
    ecalTpgRctLeadingEmEta_ = dbe->book2D("ecalTpgRctLeadingEmEta",
					  "ECAL TPG-RCT: #eta", 
					  15, -0.5, 14.5,
					  TPETABINS, TPETAMIN, TPETAMAX);
    ecalTpgRctLeadingEmEta2_ = dbe->book2D("ecalTpgRctLeadingEmEta2",
					   "ECAL TPG-RCT: #eta (2)", 
					   13, -6.5, 6.5,
					   TPETABINS, TPETAMIN, TPETAMAX);
    ecalTpgRctLeadingEmPhi_ = dbe->book2D("ecalTpgRctLeadingEmPhi",
					  "ECAL TPG-RCT: #phi", 
					  PHIBINS, PHIMIN, PHIMAX,
					  TPPHIBINS, TPPHIMIN, TPPHIMAX);
  }

}


void L1TCompare::endJob(void)
{
  if (verbose())
    std::cout << "L1TCompare: end job...." << std::endl;
  LogInfo("L1TCompare") << "analyzed " << nev_ << " events";

  if (outputFile_.size() != 0 && dbe)
    dbe->save(outputFile_);

  return;
}

void L1TCompare::analyze(const Event & e, const EventSetup & c)
{
  ++nev_;
  if (verbose()) {
    std::cout << "L1TCompare: analyze...." << std::endl;
  }

  // L1E 
  edm::Handle < L1EmParticleCollection > l1eIsoEm;
  edm::Handle < L1EmParticleCollection > l1eNonIsoEm;
  edm::Handle < L1JetParticleCollection > l1eCenJets;
  edm::Handle < L1JetParticleCollection > l1eForJets;
  edm::Handle < L1JetParticleCollection > l1eTauJets;
  edm::Handle < L1EtMissParticle > l1eEtMiss;
  // RCT
  edm::Handle < L1CaloEmCollection > em; // collection of L1CaloEmCands
  edm::Handle < L1CaloRegionCollection > rctEmRgn;

  // GCT
  edm::Handle <L1GctJetCandCollection> gctCenJets;
  edm::Handle <L1GctEmCandCollection> gctIsoEmCands;
  edm::Handle <L1GctEmCandCollection> gctNonIsoEmCands;

  try {
    e.getByLabel(rctSource_,em);
  }
  catch (...) {
    edm::LogInfo("L1TCompare") << "can't find L1CaloEmCollection with label "
			       << rctSource_.label() ;
    return;
  }

  try {
    e.getByLabel(rctSource_,rctEmRgn);
  }
  catch (...) {
    edm::LogInfo("L1TCompare") << "can't find "
			       << "L1CaloRegionCollection with label "
			       << rctSource_.label() ;
    return;
  }

  // should get rid of this try/catch?
  try {
    e.getByLabel(gctSource_.label(),"cenJets", gctCenJets);
    e.getByLabel(gctSource_.label(), "isoEm", gctIsoEmCands);
    e.getByLabel(gctSource_.label(), "nonIsoEm", gctNonIsoEmCands);
  }
  catch (...) {
    std::cerr << "L1TGCT: could not find one of the classes?" << std::endl;
    return;
  }
 

  // GCT
  if ( verbose() ) {
    for ( L1GctEmCandCollection::const_iterator iem = gctIsoEmCands->begin();
	  iem != gctIsoEmCands->end(); ++iem) {
      if ( !iem->empty() ) 
	std::cout << "GCT EM: " << iem->rank() 
		  << ", " 
		  << iem->etaIndex() << "("
	  //<< int(iem->etaIndex()&0x3)*((iem->etaIndex()&0x4)?1:-1)
		  << "), " 
		  << iem->phiIndex()
		  << std::endl;
    }
  }
  // rct phi: 0-17
  // rct eta: 0-21


  // Fill the RCT histograms

  // Regions
  RctObjectCollection rcj, rcj_iso, rcj_non_iso;
  for (L1CaloEmCollection::const_iterator iem = em->begin();
       iem != em->end(); ++iem) { 
    L1CaloRegionDetId id(false, iem->rctCrate(), iem->rctCard(), 
			 iem->rctRegion());
    RctObject h(id.gctEta(), id.gctPhi(), iem->rank());
    if ( !iem->isolated() ) 
      rcj_non_iso.push_back(h);
    else 
      rcj_iso.push_back(h);
    rcj.push_back(h);
  }
  // not so smart but ...
  std::sort(rcj.begin(), rcj.end(), RctObjectComp());
  std::sort(rcj_non_iso.begin(), rcj_non_iso.end(), RctObjectComp());
  std::sort(rcj_iso.begin(), rcj_iso.end(), RctObjectComp());
  if ( verbose() ) {
    for (RctObjectCollection::reverse_iterator ij = rcj_iso.rbegin();
	 ij != rcj_iso.rend() && ij != rcj_iso.rbegin()+8; ++ij) {
      std::cout << "RCT cj: " 
		<< ij->rank_ << ", " << ij->eta_ << ", " << ij->phi_
		<< std::endl;
    }
  }
  L1GctEmCandCollection::const_iterator lead_em = gctIsoEmCands->begin();
  if ( !lead_em->empty() ) { // equivalent to rank == 0
    rctGctLeadingIsoEmEta_->Fill(lead_em->etaIndex(), rcj_iso.rbegin()->eta_);
    rctGctLeadingIsoEmPhi_->Fill(lead_em->phiIndex(), rcj_iso.rbegin()->phi_);
    rctGctLeadingIsoEmRank_->Fill(lead_em->rank(), rcj_iso.rbegin()->rank_);
  }

  // non-isolated
  if ( verbose() ) {
    for ( L1GctEmCandCollection::const_iterator iem 
	    = gctNonIsoEmCands->begin(); iem != gctNonIsoEmCands->end(); 
	  ++iem) {
      if ( ! iem->empty() ) 
	std::cout << "GCT EM non: " << iem->rank() 
		  << ", " 
		  << iem->etaIndex() //<< "("
	  //<< int(iem->etaIndex()&0x3)*((iem->etaIndex()&0x4)?1:-1)
		  //<< ")" 
		  << ", " 
		  << iem->phiIndex()
		  << std::endl;
    }
  }
  if ( verbose() ) {
    for (RctObjectCollection::reverse_iterator ij = rcj_non_iso.rbegin();
	 ij != rcj_non_iso.rend() && ij != rcj_non_iso.rbegin()+8; ++ij) {
      std::cout << "RCT cj non: " 
		<< ij->rank_ << ", " << ij->eta_ << ", " << ij->phi_
		<< std::endl;
    }
  }
  lead_em = gctNonIsoEmCands->begin();
  if ( !lead_em->empty() ) { // equivalent to rank != 0
    rctGctLeadingNonIsoEmEta_->Fill(lead_em->etaIndex(), 
				    rcj_non_iso.rbegin()->eta_);
    rctGctLeadingNonIsoEmPhi_->Fill(lead_em->phiIndex(), 
				    rcj_non_iso.rbegin()->phi_);
    rctGctLeadingNonIsoEmRank_->Fill(lead_em->rank(), 
				     rcj_non_iso.rbegin()->rank_);
  }

  // ECAL TPG's to RCT EM 
  edm::Handle < EcalTrigPrimDigiCollection > eTP;
  try {
    e.getByLabel(ecalTpgSource_,eTP);
  }
  catch (...) {
    edm::LogInfo("L1TCompare") 
      << "can't find EcalTrigPrimCollection with label "
      << ecalTpgSource_.label() ;
    return;
  }
  RctObjectCollection ecalobs;
  for (EcalTrigPrimDigiCollection::const_iterator ieTP = eTP->begin();
       ieTP != eTP->end(); ieTP++) {
    ecalobs.push_back(RctObject(ieTP->id().ieta(),
				 ieTP->id().iphi(), 
				 ieTP->compressedEt()));
  }
  std::sort(ecalobs.begin(), ecalobs.end(), RctObjectComp());
  if ( verbose() ) {
    for (RctObjectCollection::reverse_iterator ij = ecalobs.rbegin();
	 ij != ecalobs.rend() && ij != ecalobs.rbegin()+8; ++ij) {
      std::cout << "ECAL cj : " 
		<< ij->rank_ << ", " << ij->eta_ << ", " << ij->phi_
		<< std::endl;
    }
  }
  // abritrary cut
  if ( rcj.rbegin()->rank_ > 4 ) {
    ecalTpgRctLeadingEmEta_->Fill(rcj.rbegin()->eta_,
				  ecalobs.rbegin()->eta_);
    int e2 = (rcj.rbegin()->eta_&0x7UL)* ((rcj.rbegin()->eta_&0x8UL)?1:-1);
    ecalTpgRctLeadingEmEta2_->Fill(e2, ecalobs.rbegin()->eta_);
    ecalTpgRctLeadingEmPhi_->Fill(rcj.rbegin()->phi_, ecalobs.rbegin()->phi_);
    ecalTpgRctLeadingEmRank_->Fill(rcj.rbegin()->rank_,
				   ecalobs.rbegin()->rank_);
  }
  if ( verbose() ) {
    int seta = rcj.rbegin()->eta_;
    seta = (seta&0x7UL)*(seta&0x8?-1:1);
    std::cout << "ZZ: " 
	      << rcj.rbegin()->eta_ << " "
	      << rcj.rbegin()->phi_ << " "
	      << rcj.rbegin()->rank_ << " "
	      << (++rcj.rbegin())->rank_<< " "
	      << ecalobs.rbegin()->eta_ << " "
	      << ecalobs.rbegin()->phi_ << " "
	      << ecalobs.rbegin()->rank_ << " "
	      << (++ecalobs.rbegin())->rank_<< " "
	      << seta << " "
	      << std::endl;
  }



}
