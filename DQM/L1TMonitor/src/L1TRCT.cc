/*
 * \file L1TRCT.cc
 *
 * $Date: 2012/04/04 09:56:36 $
 * $Revision: 1.22 $
 * \author P. Wittich
 *
 */

#include "DQM/L1TMonitor/interface/L1TRCT.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"

// GCT and RCT data formats
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DQMServices/Core/interface/DQMStore.h"




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
   rctSource_( ps.getParameter< InputTag >("rctSource") ),
   filterTriggerType_ (ps.getParameter< int >("filterTriggerType"))
{

  // verbosity switch
  verbose_ = ps.getUntrackedParameter < bool > ("verbose", false);

  if (verbose_)
    std::cout << "L1TRCT: constructor...." << std::endl;


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
    dbe->setCurrentFolder("L1T/L1TRCT");
  }


}

L1TRCT::~L1TRCT()
{
}

void L1TRCT::beginJob(void)
{


  nev_ = 0;

  // get hold of back-end interface
  DQMStore *dbe = 0;
  dbe = Service < DQMStore > ().operator->();

  if (dbe) {
    dbe->setCurrentFolder("L1T/L1TRCT");
    dbe->rmdir("L1T/L1TRCT");
  }


  if (dbe) {
    dbe->setCurrentFolder("L1T/L1TRCT");

    triggerType_ =
      dbe->book1D("TriggerType", "TriggerType", 17, -0.5, 16.5);

    rctIsoEmEtEtaPhi_ =
	dbe->book2D("RctEmIsoEmEtEtaPhi", "ISO EM E_{T}", ETABINS, ETAMIN,
		    ETAMAX, PHIBINS, PHIMIN, PHIMAX);
    rctIsoEmOccEtaPhi_ =
	dbe->book2D("RctEmIsoEmOccEtaPhi", "ISO EM OCCUPANCY", ETABINS,
		    ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);
    rctNonIsoEmEtEtaPhi_ =
	dbe->book2D("RctEmNonIsoEmEtEtaPhi", "NON-ISO EM E_{T}", ETABINS,
		    ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);
    rctNonIsoEmOccEtaPhi_ =
	dbe->book2D("RctEmNonIsoEmOccEtaPhi", "NON-ISO EM OCCUPANCY",
		    ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    // global regions
    rctRegionsEtEtaPhi_ =
	dbe->book2D("RctRegionsEtEtaPhi", "REGION E_{T}", ETABINS, ETAMIN,
		    ETAMAX, PHIBINS, PHIMIN, PHIMAX);
    rctRegionsOccEtaPhi_ =
	dbe->book2D("RctRegionsOccEtaPhi", "REGION OCCUPANCY", ETABINS,
		    ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctOverFlowEtaPhi_ =
	dbe->book2D("RctBitOverFlowEtaPhi", "OVER FLOW OCCUPANCY", ETABINS,
		    ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctTauVetoEtaPhi_ =
	dbe->book2D("RctBitTauVetoEtaPhi", "TAU VETO OCCUPANCY", ETABINS,
		    ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctMipEtaPhi_ =
	dbe->book2D("RctBitMipEtaPhi", "MIP OCCUPANCY", ETABINS,
		    ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctQuietEtaPhi_ =
	dbe->book2D("RctBitQuietEtaPhi", "QUIET OCCUPANCY", ETABINS,
		    ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctHfPlusTauEtaPhi_ =
	dbe->book2D("RctBitHfPlusTauEtaPhi", "HF plus Tau OCCUPANCY", ETABINS,
		    ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    // local regions
/*
    const int nlocphibins = 2; 
    const float locphimin = -0.5;
    const float locphimax = 1.5;
    const int nlocetabins = 11;
    const float locetamin = -0.5;
    const float locetamax = 10.5;
    rctRegionsLocalEtEtaPhi_ =
	dbe->book2D("RctRegionsLocalEtEtaPhi", "REGION E_{T} (Local)", 
		    nlocetabins, locetamin, locetamax,
		    nlocphibins, locphimin, locphimax);
    rctRegionsLocalOccEtaPhi_ =
	dbe->book2D("RctRegionsLocalOccEtaPhi", "REGION OCCUPANCY (Local)", 
		    nlocetabins, locetamin, locetamax,
		    nlocphibins, locphimin, locphimax);
    rctTauVetoLocalEtaPhi_ =
	dbe->book2D("RctTauLocalVetoEtaPhi", "TAU VETO OCCUPANCY (Local)",
		    nlocetabins, locetamin, locetamax,
		    nlocphibins, locphimin, locphimax);
*/
    // rank histos
    rctRegionRank_ =
	dbe->book1D("RctRegionRank", "REGION RANK", R10BINS, R10MIN,
		    R10MAX);
    rctIsoEmRank_ =
	dbe->book1D("RctEmIsoEmRank", "ISO EM RANK", R6BINS, R6MIN, R6MAX);
    rctNonIsoEmRank_ =
	dbe->book1D("RctEmNonIsoEmRank", "NON-ISO EM RANK", R6BINS, R6MIN,
		    R6MAX);
    // hw coordinates
//    rctEmCardRegion_ = dbe->book1D("rctEmCardRegion", "Em Card * Region",
//				   256, -127.5, 127.5);

    // bx histos
    rctRegionBx_ = dbe->book1D("RctRegionBx", "Region BX", 256, -0.5, 4095.5);
    rctEmBx_ = dbe->book1D("RctEmBx", "EM BX", 256, -0.5, 4095.5);

    

  }
}


void L1TRCT::endJob(void)
{
  if (verbose_)
    std::cout << "L1TRCT: end job...." << std::endl;
  LogInfo("EndJob") << "analyzed " << nev_ << " events";

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

  // filter according trigger type
  //  enum ExperimentType {
  //        Undefined          =  0,
  //        PhysicsTrigger     =  1,
  //        CalibrationTrigger =  2,
  //        RandomTrigger      =  3,
  //        Reserved           =  4,
  //        TracedEvent        =  5,
  //        TestTrigger        =  6,
  //        ErrorTrigger       = 15

  // fill a histogram with the trigger type, for normalization fill also last bin
  // ErrorTrigger + 1
  double triggerType = static_cast<double> (e.experimentType()) + 0.001;
  double triggerTypeLast = static_cast<double> (edm::EventAuxiliary::ExperimentType::ErrorTrigger)
                          + 0.001;
  triggerType_->Fill(triggerType);
  triggerType_->Fill(triggerTypeLast + 1);

  // filter only if trigger type is greater than 0, negative values disable filtering
  if (filterTriggerType_ >= 0) {

      // now filter, for real data only
      if (e.isRealData()) {
          if (!(e.experimentType() == filterTriggerType_)) {

              edm::LogInfo("L1TRCT") << "\n Event of TriggerType "
                      << e.experimentType() << " rejected" << std::endl;
              return;

          }
      }

  }

  // Get the RCT digis
  edm::Handle < L1CaloEmCollection > em;
  edm::Handle < L1CaloRegionCollection > rgn;

  bool doEm = true;
  bool doHd = true;

  e.getByLabel(rctSource_,rgn);
 
  if (!rgn.isValid()) {
    edm::LogInfo("DataNotFound") << "can't find L1CaloRegionCollection with label "
			       << rctSource_.label() ;
    doHd = false;
  }

  if ( doHd ) {
    // Fill the RCT histograms

    // Regions
    for (L1CaloRegionCollection::const_iterator ireg = rgn->begin();
	 ireg != rgn->end(); ireg++) {
      if(ireg->et()>0)
      {
      rctRegionRank_->Fill(ireg->et());
      if(ireg->et()>5){
	rctRegionsOccEtaPhi_->Fill(ireg->gctEta(), ireg->gctPhi());
      }
      rctRegionsEtEtaPhi_->Fill(ireg->gctEta(), ireg->gctPhi(), ireg->et());
//      rctTauVetoEtaPhi_->Fill(ireg->gctEta(), ireg->gctPhi(),
//			      ireg->tauVeto());

      // now do local coordinate eta and phi
//      rctRegionsLocalOccEtaPhi_->Fill(ireg->rctEta(), ireg->rctPhi());
//      rctRegionsLocalEtEtaPhi_->Fill(ireg->rctEta(), ireg->rctPhi(), 
//				     ireg->et());
//      rctTauVetoLocalEtaPhi_->Fill(ireg->rctEta(), ireg->rctPhi(),
//				   ireg->tauVeto());
      rctRegionBx_->Fill(ireg->bx());
      }

    if(ireg->overFlow())  rctOverFlowEtaPhi_ ->Fill(ireg->gctEta(), ireg->gctPhi());
    if(ireg->tauVeto())   rctTauVetoEtaPhi_  ->Fill(ireg->gctEta(), ireg->gctPhi());
    if(ireg->mip())       rctMipEtaPhi_      ->Fill(ireg->gctEta(), ireg->gctPhi());
    if(ireg->quiet())     rctQuietEtaPhi_    ->Fill(ireg->gctEta(), ireg->gctPhi());
    if(ireg->fineGrain()) rctHfPlusTauEtaPhi_->Fill(ireg->gctEta(), ireg->gctPhi()); 
    
    }
  }

  
  e.getByLabel(rctSource_,em);
  
  if (!em.isValid()) {
    edm::LogInfo("DataNotFound") << "can't find L1CaloEmCollection with label "
			       << rctSource_.label() ;
    doEm = false;
  }
  if ( ! doEm ) return;
  // Isolated and non-isolated EM
  for (L1CaloEmCollection::const_iterator iem = em->begin();
       iem != em->end(); iem++) {
    
 //   rctEmCardRegion_->Fill((iem->rctRegion()==0?1:-1)*(iem->rctCard()));

    if (iem->isolated()) {
      if(iem->rank()>0)
      {
      rctIsoEmRank_->Fill(iem->rank());
      rctIsoEmEtEtaPhi_->Fill(iem->regionId().ieta(),
			      iem->regionId().iphi(), iem->rank());
      if(iem->rank()>10){
	rctIsoEmOccEtaPhi_->Fill(iem->regionId().ieta(),
				 iem->regionId().iphi());
      }
      rctEmBx_->Fill(iem->bx());
      }
    }
    else {
      if(iem->rank()>0)
      { 
      rctNonIsoEmRank_->Fill(iem->rank());
      rctNonIsoEmEtEtaPhi_->Fill(iem->regionId().ieta(),
				 iem->regionId().iphi(), iem->rank());
      if(iem->rank()>10){
	rctNonIsoEmOccEtaPhi_->Fill(iem->regionId().ieta(),
				    iem->regionId().iphi());
      }
      rctEmBx_->Fill(iem->bx());
      }
    }

  }

}
