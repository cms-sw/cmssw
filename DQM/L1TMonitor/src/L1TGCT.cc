/*
 * \file L1TGCT.cc
 *
 * $Date: 2007/02/22 19:43:53 $
 * $Revision: 1.6 $
 * \author J. Berryhill
 *
 *  Initial version largely stolen from GCTMonitor (wittich 2/07)
 *
 * $Log: L1TGCT.cc,v $
 * Revision 1.6  2007/02/22 19:43:53  berryhil
 *
 *
 *
 * InputTag parameters added for all modules
 *
 * Revision 1.5  2007/02/20 22:49:00  wittich
 * - change from getByType to getByLabel in ECAL TPG,
 *   and make it configurable.
 * - fix problem in the GCT with incorrect labels. Not the ultimate
 *   solution - will probably have to go to many labels.
 *
 * Revision 1.4  2007/02/19 22:49:54  wittich
 * - Add RCT monitor
 *
 * Revision 1.3  2007/02/19 22:07:26  wittich
 * - Added three monitorables to the ECAL TPG monitoring (from GCTMonitor)
 * - other minor tweaks in GCT, etc
 *
 * Revision 1.2  2007/02/19 21:11:23  wittich
 * - Updates for integrating GCT monitor.
 *   + Adapted right now only the L1E elements thereof.
 *   + added DataFormats/L1Trigger to build file.
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
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"

// L1Extra
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"


using namespace edm;
using namespace l1extra;


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


const unsigned int L1EETABINS = 22;
const float L1EETAMIN = -5;
const float L1EETAMAX = 5;

const unsigned int L1EPHIBINS = 18;
const float L1EPHIMIN = -M_PI;
const float L1EPHIMAX = M_PI;

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

// Physical bins 1 Gev - 1 TeV in 1 GeV steps
const unsigned int TEVBINS = 1001;
const float TEVMIN = -0.5;
const float TEVMAX = 1000.5;


L1TGCT::L1TGCT(const edm::ParameterSet & ps) :
  gctSource_(ps.getParameter<edm::InputTag>("gctSource"))
{

  // verbosity switch
  verbose_ = ps.getUntrackedParameter < bool > ("verbose", false);

  if (verbose_)
    std::cout << "L1TGCT: constructor...." << std::endl;

  logFile_.open("L1TGCT.log");

  dbe = NULL;
  if (ps.getUntrackedParameter < bool > ("DaqMonitorBEInterface", false)) {
    dbe = edm::Service < DaqMonitorBEInterface > ().operator->();
    dbe->setVerbose(0);
  }

  monitorDaemon_ = false;
  if (ps.getUntrackedParameter < bool > ("MonitorDaemon", false)) {
    edm::Service<MonitorDaemon> daemon;
    daemon.operator->();
    monitorDaemon_ = true;
  }

  outputFile_ = ps.getUntrackedParameter < std::string > ("outputFile", "");
  if (outputFile_.size() != 0) {
    std::cout << "L1T Monitoring histograms will be saved to "
	      << outputFile_ << std::endl;
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
    dbe->setCurrentFolder("L1TMonitor/L1TGCT");
  }


}

L1TGCT::~L1TGCT()
{
}

void L1TGCT::beginJob(const edm::EventSetup & c)
{

  nev_ = 0;

  // get hold of back-end interface
  DaqMonitorBEInterface *dbe = 0;
  dbe = edm::Service < DaqMonitorBEInterface > ().operator->();

  if (dbe) {
    dbe->setCurrentFolder("L1TMonitor/L1TGCT");
    dbe->rmdir("L1TMonitor/L1TGCT");
  }


  if (dbe) {
    dbe->setCurrentFolder("L1TMonitor/L1TGCT");

    // Book L1Extra histograms
    //dbe->setCurrentFolder("L1Extra"); // Add subfolder

    l1ExtraCenJetsEtEtaPhi_ =
	dbe->book2D("L1ExtraCenJetsEtEtaPhi", "CENTRAL JET E_{T}",
		    L1EPHIBINS, L1EPHIMIN, L1EPHIMAX, L1EETABINS,
		    L1EETAMIN, L1EETAMAX);
    l1ExtraForJetsEtEtaPhi_ =
	dbe->book2D("L1ExtraForJetsEtEtaPhi", "FORWARD JET E_{T}",
		    L1EPHIBINS, L1EPHIMIN, L1EPHIMAX, L1EETABINS,
		    L1EETAMIN, L1EETAMAX);
    l1ExtraTauJetsEtEtaPhi_ =
	dbe->book2D("L1ExtraTauJetsEtEtaPhi", "TAU JET E_{T}", L1EPHIBINS,
		    L1EPHIMIN, L1EPHIMAX, L1EETABINS, L1EETAMIN,
		    L1EETAMAX);
    l1ExtraIsoEmEtEtaPhi_ =
	dbe->book2D("L1ExtraIsoEmEtEtaPhi", "ISO EM E_{T}", L1EPHIBINS,
		    L1EPHIMIN, L1EPHIMAX, L1EETABINS, L1EETAMIN,
		    L1EETAMAX);
    l1ExtraNonIsoEmEtEtaPhi_ =
	dbe->book2D("L1ExtraNonIsoEmEtEtaPhi", "NON-ISO EM E_{T}",
		    L1EPHIBINS, L1EPHIMIN, L1EPHIMAX, L1EETABINS,
		    L1EETAMIN, L1EETAMAX);

    l1ExtraCenJetsOccEtaPhi_ =
	dbe->book2D("L1ExtraCenJetsOccEtaPhi", "CENTRAL JET OCCUPANCY",
		    L1EPHIBINS, L1EPHIMIN, L1EPHIMAX, L1EETABINS,
		    L1EETAMIN, L1EETAMAX);
    l1ExtraForJetsOccEtaPhi_ =
	dbe->book2D("L1ExtraForJetsOccEtaPhi", "FORWARD JET OCCUPANCY",
		    L1EPHIBINS, L1EPHIMIN, L1EPHIMAX, L1EETABINS,
		    L1EETAMIN, L1EETAMAX);
    l1ExtraTauJetsOccEtaPhi_ =
	dbe->book2D("L1ExtraTauJetsOccEtaPhi", "TAU JET OCCUPANCY",
		    L1EPHIBINS, L1EPHIMIN, L1EPHIMAX, L1EETABINS,
		    L1EETAMIN, L1EETAMAX);
    l1ExtraIsoEmOccEtaPhi_ =
	dbe->book2D("L1ExtraIsoEmOccEtaPhi", "ISO EM OCCUPANCY",
		    L1EPHIBINS, L1EPHIMIN, L1EPHIMAX, L1EETABINS,
		    L1EETAMIN, L1EETAMAX);
    l1ExtraNonIsoEmOccEtaPhi_ =
	dbe->book2D("L1ExtraNonIsoEmOccEtaPhi", "NON-ISO EM OCCUPANCY",
		    L1EPHIBINS, L1EPHIMIN, L1EPHIMAX, L1EETABINS,
		    L1EETAMIN, L1EETAMAX);

    l1ExtraCenJetsRank_ =
	dbe->book1D("L1ExtraCenJetsRank", "CENTRAL JET RANK", TEVBINS,
		    TEVMIN, TEVMAX);
    l1ExtraForJetsRank_ =
	dbe->book1D("L1ExtraForJetsRank", "FORWARD JET RANK", TEVBINS,
		    TEVMIN, TEVMAX);
    l1ExtraTauJetsRank_ =
	dbe->book1D("L1ExtraTauJetsRank", "TAU JET RANK", TEVBINS, TEVMIN,
		    TEVMAX);
    l1ExtraIsoEmRank_ =
	dbe->book1D("L1ExtraIsoEmRank", "ISO EM RANK", TEVBINS, TEVMIN,
		    TEVMAX);
    l1ExtraNonIsoEmRank_ =
	dbe->book1D("L1ExtraNonIsoEmRank", "NON-ISO EM RANK", TEVBINS,
		    TEVMIN, TEVMAX);

    l1ExtraEtMiss_ =
	dbe->book1D("L1ExtraEtMiss", "MISSING E_{T}", TEVBINS, TEVMIN,
		    TEVMAX);
    l1ExtraEtMissPhi_ =
	dbe->book1D("L1ExtraEtMissPhi", "MISSING E_{T} #phi", METPHIBINS,
		    L1EPHIMIN, L1EPHIMAX);
    l1ExtraEtTotal_ =
	dbe->book1D("L1ExtraEtTotal", "TOTAL E_{T}", TEVBINS, TEVMIN,
		    TEVMAX);
    l1ExtraEtHad_ =
	dbe->book1D("L1ExtraEtHad", "TOTAL HAD E_{T}", TEVBINS, TEVMIN,
		    TEVMAX);
  }
}


void L1TGCT::endJob(void)
{
  if (verbose_)
    std::cout << "L1TGCT: end job...." << std::endl;
  edm::LogInfo("L1TGCT") << "analyzed " << nev_ << " events";

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

  // L1 Extra information - these are in physics quantities
  // get the L1Extra collections
  edm::Handle < L1EmParticleCollection > l1eIsoEm;
  edm::Handle < L1EmParticleCollection > l1eNonIsoEm;
  edm::Handle < L1JetParticleCollection > l1eCenJets;
  edm::Handle < L1JetParticleCollection > l1eForJets;
  edm::Handle < L1JetParticleCollection > l1eTauJets;
  edm::Handle < L1EtMissParticle > l1eEtMiss;

  // should get rid of this try/catch?
  try {
    e.getByLabel(gctSource_.label(), "Isolated", l1eIsoEm);
    e.getByLabel(gctSource_.label(), "NonIsolated", l1eNonIsoEm);
    e.getByLabel(gctSource_.label(), "Central", l1eCenJets);
    e.getByLabel(gctSource_.label(), "Forward", l1eForJets);
    e.getByLabel(gctSource_.label(), "Tau", l1eTauJets);

    e.getByLabel(gctSource_, l1eEtMiss);
  }
  catch (...) {
    edm::LogInfo("L1TGCT") << " Could not find one of the requested data "
      "elements." ;
    return;
  }


  // Fill the L1Extra histograms

  // Central jets
  for (L1JetParticleCollection::const_iterator cj = l1eCenJets->begin();
       cj != l1eCenJets->end(); cj++) {
    l1ExtraCenJetsEtEtaPhi_->Fill(cj->phi(), cj->eta(), cj->et());
    l1ExtraCenJetsOccEtaPhi_->Fill(cj->phi(), cj->eta());
    l1ExtraCenJetsRank_->Fill(cj->et());
  }

  // Forward jets
  for (L1JetParticleCollection::const_iterator fj = l1eForJets->begin();
       fj != l1eForJets->end(); fj++) {
    l1ExtraForJetsEtEtaPhi_->Fill(fj->phi(), fj->eta(), fj->et());
    l1ExtraForJetsOccEtaPhi_->Fill(fj->phi(), fj->eta());
    l1ExtraForJetsRank_->Fill(fj->et());
  }

  // Tau jets
  for (L1JetParticleCollection::const_iterator tj = l1eTauJets->begin();
       tj != l1eTauJets->end(); tj++) {
    l1ExtraTauJetsEtEtaPhi_->Fill(tj->phi(), tj->eta(), tj->et());
    l1ExtraTauJetsOccEtaPhi_->Fill(tj->phi(), tj->eta());
    l1ExtraTauJetsRank_->Fill(tj->et());
  }

  // Isolated EM
  for (L1EmParticleCollection::const_iterator ie = l1eIsoEm->begin();
       ie != l1eIsoEm->end(); ie++) {
    l1ExtraIsoEmEtEtaPhi_->Fill(ie->phi(), ie->eta(), ie->et());
    l1ExtraIsoEmOccEtaPhi_->Fill(ie->phi(), ie->eta());
    l1ExtraIsoEmRank_->Fill(ie->et());
  }

  // Non-isolated EM
  for (L1EmParticleCollection::const_iterator ne = l1eNonIsoEm->begin();
       ne != l1eNonIsoEm->end(); ne++) {
    l1ExtraNonIsoEmEtEtaPhi_->Fill(ne->phi(), ne->eta(), ne->et());
    l1ExtraNonIsoEmOccEtaPhi_->Fill(ne->phi(), ne->eta());
    l1ExtraNonIsoEmRank_->Fill(ne->et());
  }

  // Energy sums
  l1ExtraEtMiss_->Fill(l1eEtMiss->et());
  l1ExtraEtMissPhi_->Fill(l1eEtMiss->phi());
  l1ExtraEtTotal_->Fill(l1eEtMiss->etTotal());
  l1ExtraEtHad_->Fill(l1eEtMiss->etHad());






}
