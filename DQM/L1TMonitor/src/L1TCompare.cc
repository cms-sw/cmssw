/*
 * \file L1TCompare.cc
 * $Id: L1TCompare.cc,v 1.14 2009/11/19 14:36:48 puigh Exp $
 * \author P. Wittich
 * \brief Compare different parts of the trigger chain (e.g., RCT-GCT )
 * $Log: L1TCompare.cc,v $
 * Revision 1.14  2009/11/19 14:36:48  puigh
 * modify beginJob
 *
 * Revision 1.13  2008/03/20 19:38:25  berryhil
 *
 *
 * organized message logger
 *
 * Revision 1.12  2008/03/14 20:35:46  berryhil
 *
 *
 * stripped out obsolete parameter settings
 *
 * rpc tpg restored with correct dn access and dbe handling
 *
 * Revision 1.11  2008/03/12 17:24:24  berryhil
 *
 *
 * eliminated log files, truncated HCALTPGXana histo output
 *
 * Revision 1.10  2008/03/01 00:40:00  lat
 * DQM core migration.
 *
 * Revision 1.9  2008/01/22 18:56:01  muzaffar
 * include cleanup. Only for cc/cpp files
 *
 * Revision 1.8  2007/12/21 20:04:50  wsun
 * Migrated L1EtMissParticle -> L1EtMissParticleCollection.
 *
 * Revision 1.7  2007/12/21 17:41:20  berryhil
 *
 *
 * try/catch removal
 *
 * Revision 1.6  2007/11/19 15:08:22  lorenzo
 * changed top folder name
 *
 * Revision 1.5  2007/09/27 22:58:15  ratnik
 * QA campaign: fixes to compensate includes cleanup in  DataFormats/L1Trigger
 *
 * Revision 1.4  2007/07/19 18:05:06  berryhil
 *
 *
 * L1CaloRegionDetId dataformat migration for L1TCompare
 *
 * Revision 1.3  2007/06/13 11:33:39  wittich
 * add axis titles
 *
 * Revision 1.2  2007/06/08 08:37:43  wittich
 * Add ECAL TP - RCT comparisons. Lingering problems with
 * mismatches right now - still needs work.
 *
 * Revision 1.1  2007/06/06 14:55:51  wittich
 * compare within trigger subsystems
 *
 */

#include "DQM/L1TMonitor/interface/L1TCompare.h"

// GCT and RCT data formats
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloRegionDetId.h"

// L1Extra
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticleFwd.h"

// Ecal
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include "DQMServices/Core/interface/DQMStore.h"

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


  dbe = NULL;
  if (ps.getUntrackedParameter < bool > ("DQMStore", false)) {
    dbe = Service < DQMStore > ().operator->();
    dbe->setVerbose(0);
  }

  outputFile_ =
      ps.getUntrackedParameter < std::string > ("outputFile", "");
  if (outputFile_.size() != 0) {
    std::
	cout << "L1T Monitoring histograms will be saved to " <<
	outputFile_.c_str() << std::endl;
  }

  bool disable =
      ps.getUntrackedParameter < bool > ("disableROOToutput", false);
  if (disable) {
    outputFile_ = "";
  }


  if (dbe != NULL) {
    dbe->setCurrentFolder("L1T/Compare");
  }


}

L1TCompare::~L1TCompare()
{
}

void L1TCompare::beginJob(void)
{

  nev_ = 0;

  // get hold of back-end interface
  DQMStore *dbe = 0;
  dbe = Service < DQMStore > ().operator->();

  if (dbe) {
    dbe->setCurrentFolder("L1T/Compare");
    dbe->rmdir("L1T/Compare");
  }


  if (dbe) {
    dbe->setCurrentFolder("L1T/Compare");
    
    // -------------------------------------------
    // RCT-GCT
    // -------------------------------------------
    // Isolated
    rctGctLeadingIsoEmRank_ = dbe->book2D("rctGctLeadingIsoEmRank",
				       "RCT-GCT: rank", R6BINS, R6MIN, R6MAX,
				       R6BINS, R6MIN, R6MAX);
    rctGctLeadingIsoEmRank_->setAxisTitle(std::string("gct"), 1);
    rctGctLeadingIsoEmRank_->setAxisTitle(std::string("rct"), 2);
    rctGctLeadingIsoEmEta_ = dbe->book2D("rctGctLeadingIsoEmEta",
				      "RCT-GCT: #eta", ETABINS, ETAMIN, ETAMAX,
				       ETABINS, ETAMIN, ETAMAX);
    rctGctLeadingIsoEmEta_->setAxisTitle(std::string("gct"), 1);
    rctGctLeadingIsoEmEta_->setAxisTitle(std::string("rct"), 2);

    rctGctLeadingIsoEmPhi_ = dbe->book2D("rctGctLeadingIsoEmPhi",
				      "RCT-GCT: #phi", PHIBINS, PHIMIN, PHIMAX,
				       PHIBINS, PHIMIN, PHIMAX);
    rctGctLeadingIsoEmPhi_->setAxisTitle(std::string("gct"), 1);
    rctGctLeadingIsoEmPhi_->setAxisTitle(std::string("rct"), 2);
    // non-Isolated
    rctGctLeadingNonIsoEmRank_ = dbe->book2D("rctGctLeadingNonIsoEmRank",
				       "RCT-GCT: rank", R6BINS, R6MIN, R6MAX,
				       R6BINS, R6MIN, R6MAX);
    rctGctLeadingNonIsoEmRank_->setAxisTitle(std::string("gct"), 1);
    rctGctLeadingNonIsoEmRank_->setAxisTitle(std::string("rct"), 2);

    rctGctLeadingNonIsoEmEta_ = dbe->book2D("rctGctLeadingNonIsoEmEta",
				      "RCT-GCT: #eta", ETABINS, ETAMIN, ETAMAX,
				       ETABINS, ETAMIN, ETAMAX);
    rctGctLeadingNonIsoEmEta_->setAxisTitle(std::string("gct"), 1);
    rctGctLeadingNonIsoEmEta_->setAxisTitle(std::string("rct"), 2);

    rctGctLeadingNonIsoEmPhi_ = dbe->book2D("rctGctLeadingNonIsoEmPhi",
				      "RCT-GCT: #phi", PHIBINS, PHIMIN, PHIMAX,
				       PHIBINS, PHIMIN, PHIMAX);
    rctGctLeadingNonIsoEmPhi_->setAxisTitle(std::string("gct"), 1);
    rctGctLeadingNonIsoEmPhi_->setAxisTitle(std::string("rct"), 2);
    // -------------------------------------------
    // ECAL TPG - RCT
    // -------------------------------------------
    ecalTpgRctLeadingEmRank_ = dbe->book2D("ecalTpgRctLeadingEmRank",
					   "ECAL TPG-RCT: rank", 
					   R6BINS, R6MIN, R6MAX,
					   R6BINS, R6MIN, R6MAX);
    ecalTpgRctLeadingEmRank_->setAxisTitle(std::string("rct"), 1);
    ecalTpgRctLeadingEmRank_->setAxisTitle(std::string("ecal tp"), 2);

    ecalTpgRctLeadingEmEta_ = dbe->book2D("ecalTpgRctLeadingEmEta",
					  "ECAL TPG-RCT: #eta", 
					  15, -0.5, 14.5,
					  TPETABINS, TPETAMIN, TPETAMAX);
    ecalTpgRctLeadingEmEta_->setAxisTitle(std::string("rct"), 1);
    ecalTpgRctLeadingEmEta_->setAxisTitle(std::string("ecal tp"), 2);
    ecalTpgRctLeadingEmEta2_ = dbe->book2D("ecalTpgRctLeadingEmEta2",
					   "ECAL TPG-RCT: #eta (2)", 
					   13, -6.5, 6.5,
					   TPETABINS, TPETAMIN, TPETAMAX);
    ecalTpgRctLeadingEmEta2_->setAxisTitle(std::string("rct"), 1);
    ecalTpgRctLeadingEmEta2_->setAxisTitle(std::string("ecal tp"), 2);
    ecalTpgRctLeadingEmPhi_ = dbe->book2D("ecalTpgRctLeadingEmPhi",
					  "ECAL TPG-RCT: #phi", 
					  PHIBINS, PHIMIN, PHIMAX,
					  TPPHIBINS, TPPHIMIN, TPPHIMAX);
    ecalTpgRctLeadingEmPhi_->setAxisTitle(std::string("rct"), 1);
    ecalTpgRctLeadingEmPhi_->setAxisTitle(std::string("ecal tp"), 2);
  }

}


void L1TCompare::endJob(void)
{
  if (verbose())
    std::cout << "L1TCompare: end job...." << std::endl;
  LogInfo("EndJob") << "analyzed " << nev_ << " events";

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
  //  edm::Handle < L1EtMissParticle > l1eEtMiss;
  edm::Handle < L1EtMissParticleCollection > l1eEtMiss;
  // RCT
  edm::Handle < L1CaloEmCollection > em; // collection of L1CaloEmCands
  edm::Handle < L1CaloRegionCollection > rctEmRgn;

  // GCT
  edm::Handle <L1GctJetCandCollection> gctCenJets;
  edm::Handle <L1GctEmCandCollection> gctIsoEmCands;
  edm::Handle <L1GctEmCandCollection> gctNonIsoEmCands;

  
  e.getByLabel(rctSource_,em);
  
  if (!em.isValid()) {
    edm::LogInfo("DataNotFound") << "can't find L1CaloEmCollection with label "
			       << rctSource_.label() ;
    return;
  }

  
  e.getByLabel(rctSource_,rctEmRgn);
  
  if (!rctEmRgn.isValid()) {
    edm::LogInfo("DataNotFound") << "can't find "
			       << "L1CaloRegionCollection with label "
			       << rctSource_.label() ;
    return;
  }

  
  e.getByLabel(gctSource_.label(),"cenJets", gctCenJets);
  e.getByLabel(gctSource_.label(), "isoEm", gctIsoEmCands);
  e.getByLabel(gctSource_.label(), "nonIsoEm", gctNonIsoEmCands);
  
  if (!gctCenJets.isValid()) {
    std::cerr << "L1TGCT: could not find one of the classes?" << std::endl;
    return;
  }
 if (!gctIsoEmCands.isValid()) {
    std::cerr << "L1TGCT: could not find one of the classes?" << std::endl;
    return;
  }
 if (!gctNonIsoEmCands.isValid()) {
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
    //   L1CaloRegionDetId id(false, iem->rctCrate(), iem->rctCard(), 
    //			 iem->rctRegion());
       L1CaloRegionDetId id(iem->rctCrate(), iem->rctCard(), 
    			 iem->rctRegion());

       // RctObject h(id.gctEta(), id.gctPhi(), iem->rank());
    RctObject h(id.rctEta(), id.rctPhi(), iem->rank());
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
  e.getByLabel(ecalTpgSource_,eTP);
  
  if (!eTP.isValid()) {
    edm::LogInfo("DataNotFound") 
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
