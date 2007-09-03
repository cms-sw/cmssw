/*
 * \file L1TRCT.cc
 *
 * $Date: 2007/05/25 15:45:48 $
 * $Revision: 1.3 $
 * \author P. Wittich
 *
 */

#include "DQM/L1TMonitor/interface/L1TRCT.h"

// GCT and RCT data formats
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtSums.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"




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

const unsigned int ETABINS = 22;
const float ETAMIN = -0.5;
const float ETAMAX = 21.5;



L1TRCT::L1TRCT(const ParameterSet & ps) :
   rctSource_( ps.getParameter< InputTag >("rctSource") )

{

  // verbosity switch
  verbose_ = ps.getUntrackedParameter < bool > ("verbose", false);

  if (verbose_)
    std::cout << "L1TRCT: constructor...." << std::endl;

  logFile_.open("L1TRCT.log");

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

L1TRCT::~L1TRCT()
{
}

void L1TRCT::beginJob(const EventSetup & c)
{

  nev_ = 0;

  // get hold of back-end interface
  DaqMonitorBEInterface *dbe = 0;
  dbe = Service < DaqMonitorBEInterface > ().operator->();

  if (dbe) {
    dbe->setCurrentFolder("L1TMonitor/L1TRCT");
    dbe->rmdir("L1TMonitor/L1TRCT");
  }


  if (dbe) {
    dbe->setCurrentFolder("L1TMonitor/L1TRCT");

    rctIsoEmEtEtaPhi_ =
	dbe->book2D("RctIsoEmEtEtaPhi", "ISO EM E_{T}", PHIBINS, PHIMIN,
		    PHIMAX, ETABINS, ETAMIN, ETAMAX);
    rctIsoEmOccEtaPhi_ =
	dbe->book2D("RctIsoEmOccEtaPhi", "ISO EM OCCUPANCY", PHIBINS,
		    PHIMIN, PHIMAX, ETABINS, ETAMIN, ETAMAX);
    rctNonIsoEmEtEtaPhi_ =
	dbe->book2D("RctNonIsoEmEtEtaPhi", "NON-ISO EM E_{T}", PHIBINS,
		    PHIMIN, PHIMAX, ETABINS, ETAMIN, ETAMAX);
    rctNonIsoEmOccEtaPhi_ =
	dbe->book2D("RctNonIsoEmOccEtaPhi", "NON-ISO EM OCCUPANCY",
		    PHIBINS, PHIMIN, PHIMAX, ETABINS, ETAMIN, ETAMAX);

    // global regions
    rctRegionsEtEtaPhi_ =
	dbe->book2D("RctRegionsEtEtaPhi", "REGION E_{T}", PHIBINS, PHIMIN,
		    PHIMAX, ETABINS, ETAMIN, ETAMAX);
    rctRegionsOccEtaPhi_ =
	dbe->book2D("RctRegionsOccEtaPhi", "REGION OCCUPANCY", PHIBINS,
		    PHIMIN, PHIMAX, ETABINS, ETAMIN, ETAMAX);
    rctTauVetoEtaPhi_ =
	dbe->book2D("RctTauVetoEtaPhi", "TAU VETO OCCUPANCY", PHIBINS,
		    PHIMIN, PHIMAX, ETABINS, ETAMIN, ETAMAX);

    // local regions
    const int nlocphibins = 2; 
    const float locphimin = -0.5;
    const float locphimax = 1.5;
    const int nlocetabins = 11;
    const float locetamin = -0.5;
    const float locetamax = 10.5;
    rctRegionsLocalEtEtaPhi_ =
	dbe->book2D("RctRegionsEtEtaPhi", "REGION E_{T}", 
		    nlocphibins, locphimin, locphimax,
		    nlocetabins, locetamin, locetamax);
    rctRegionsLocalOccEtaPhi_ =
	dbe->book2D("RctRegionsOccEtaPhi", "REGION OCCUPANCY", 
		    nlocphibins, locphimin, locphimax,
		    nlocetabins, locetamin, locetamax);
    rctTauVetoLocalEtaPhi_ =
	dbe->book2D("RctTauVetoEtaPhi", "TAU VETO OCCUPANCY",
		    nlocphibins, locphimin, locphimax,
		    nlocetabins, locetamin, locetamax);

    // rank histos
    rctRegionRank_ =
	dbe->book1D("RctRegionRank", "REGION RANK", R10BINS, R10MIN,
		    R10MAX);
    rctIsoEmRank_ =
	dbe->book1D("RctIsoEmRank", "ISO EM RANK", R6BINS, R6MIN, R6MAX);
    rctNonIsoEmRank_ =
	dbe->book1D("RctNonIsoEmRank", "NON-ISO EM RANK", R6BINS, R6MIN,
		    R6MAX);
    // hw coordinates
    rctEmCardRegion_ = dbe->book1D("rctEmCardRegion", "Em Card * Region",
				   256, -127.5, 127.5);

    // bx histos
    rctRegionBx_ = dbe->book1D("RctRegionBx", "Region BX", 256, -0.5, 4095.5);
    rctEmBx_ = dbe->book1D("RctEmBx", "EM BX", 256, -0.5, 4095.5);

    

  }
}


void L1TRCT::endJob(void)
{
  if (verbose_)
    std::cout << "L1TRCT: end job...." << std::endl;
  LogInfo("L1TRCT") << "analyzed " << nev_ << " events";

  if (outputFile_.size() != 0 && dbe)
    dbe->save(outputFile_);

  return;
}

void L1TRCT::analyze(const Event & e, const EventSetup & c)
{
  nev_++;
  if (verbose_) {
    std::cout << "L1TRCT: analyze...." << std::endl;
  }

  // Get the RCT digis
  edm::Handle < L1CaloEmCollection > em;
  edm::Handle < L1CaloRegionCollection > rgn;

  // need to change to getByLabel
 
  try {
  e.getByLabel(rctSource_,em);
  }
  catch (...) {
    edm::LogInfo("L1TRCT") << "can't find L1CaloEmCollection with label "
			       << rctSource_.label() ;
    return;
  }

  try {
  e.getByLabel(rctSource_,rgn);
  }
  catch (...) {
    edm::LogInfo("L1TRCT") << "can't find L1CaloRegionCollection with label "
			       << rctSource_.label() ;
    return;
  }


  // Fill the RCT histograms

  // Regions
  for (L1CaloRegionCollection::const_iterator ireg = rgn->begin();
       ireg != rgn->end(); ireg++) {
    rctRegionsOccEtaPhi_->Fill(ireg->gctPhi(), ireg->gctEta());
    rctRegionsEtEtaPhi_->Fill(ireg->gctPhi(), ireg->gctEta(), ireg->et());
    rctRegionRank_->Fill(ireg->et());
    rctTauVetoEtaPhi_->Fill(ireg->gctPhi(), ireg->gctEta(),
			    ireg->tauVeto());

    // now do local coordinate eta and phi
    rctRegionsLocalOccEtaPhi_->Fill(ireg->rctPhi(), ireg->rctEta());
    rctRegionsLocalEtEtaPhi_->Fill(ireg->rctPhi(), ireg->rctEta(), ireg->et());
    rctTauVetoLocalEtaPhi_->Fill(ireg->rctPhi(), ireg->rctEta(),
			    ireg->tauVeto());
    
    rctRegionBx_->Fill(ireg->bx());
    
  }

  // Isolated and non-isolated EM
  for (L1CaloEmCollection::const_iterator iem = em->begin();
       iem != em->end(); iem++) {
    
    rctEmCardRegion_->Fill((iem->rctRegion()==0?1:-1)*(iem->rctCard()));

    if (iem->isolated()) {
      rctIsoEmRank_->Fill(iem->rank());
      rctIsoEmEtEtaPhi_->Fill(iem->regionId().iphi(),
			      iem->regionId().ieta(), iem->rank());
      rctIsoEmOccEtaPhi_->Fill(iem->regionId().iphi(),
			       iem->regionId().ieta());

    }
    else {
      rctNonIsoEmRank_->Fill(iem->rank());
      rctNonIsoEmEtEtaPhi_->Fill(iem->regionId().iphi(),
				 iem->regionId().ieta(), iem->rank());
      rctNonIsoEmOccEtaPhi_->Fill(iem->regionId().iphi(),
				  iem->regionId().ieta());
    }
    rctEmBx_->Fill(iem->bx());

  }

}
