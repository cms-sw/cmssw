/*
 * \file L1TRCT.cc
 *
 * \author P. Wittich
 *
 */

#include "DQM/L1TMonitor/interface/L1TRCT.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"

//DQMStore
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

const unsigned int ETABINS = 22;
const float ETAMIN = -0.5;
const float ETAMAX = 21.5;



L1TRCT::L1TRCT(const ParameterSet & ps) :
   rctSource_L1CRCollection_( consumes<L1CaloRegionCollection>(ps.getParameter< InputTag >("rctSource") )),
   rctSource_L1CEMCollection_( consumes<L1CaloEmCollection>(ps.getParameter< InputTag >("rctSource") )),
   filterTriggerType_ (ps.getParameter< int >("filterTriggerType"))
{

  // verbosity switch
  verbose_ = ps.getUntrackedParameter < bool > ("verbose", false);

  if (verbose_)
    std::cout << "L1TRCT: constructor...." << std::endl;

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
}

L1TRCT::~L1TRCT()
{
}



void L1TRCT::dqmBeginRun(const edm::Run& r, const edm::EventSetup& c){
  //runId_->Fill(r.id().run());
  //
}

void L1TRCT::beginLuminosityBlock(const edm::LuminosityBlock& l, const edm::EventSetup& c){
  //
  //lumisecId_->Fill(l.id().luminosityBlock());
}


void L1TRCT::bookHistograms(DQMStore::IBooker &ibooker, edm::Run const&, edm::EventSetup const&)
{
  nev_ = 0;
  
  ibooker.setCurrentFolder("L1T/L1TRCT");

  runId_=ibooker.bookInt("iRun");
  runId_->Fill(-1);
  lumisecId_=ibooker.bookInt("lumiSection");
  lumisecId_->Fill(-1);
  

  triggerType_ = ibooker.book1D("TriggerType", "TriggerType", 17, -0.5, 16.5);

  rctIsoEmEtEtaPhi_ =	ibooker.book2D("RctEmIsoEmEtEtaPhi", "ISO EM E_{T}", ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);
  rctIsoEmOccEtaPhi_ = ibooker.book2D("RctEmIsoEmOccEtaPhi", "ISO EM OCCUPANCY", ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);
  rctNonIsoEmEtEtaPhi_ = ibooker.book2D("RctEmNonIsoEmEtEtaPhi", "NON-ISO EM E_{T}", ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);
  rctNonIsoEmOccEtaPhi_ = ibooker.book2D("RctEmNonIsoEmOccEtaPhi", "NON-ISO EM OCCUPANCY",ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    // global regions
  rctRegionsEtEtaPhi_ = ibooker.book2D("RctRegionsEtEtaPhi", "REGION E_{T}", ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);
  rctRegionsOccEtaPhi_ = ibooker.book2D("RctRegionsOccEtaPhi", "REGION OCCUPANCY", ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

  rctOverFlowEtaPhi_ = ibooker.book2D("RctBitOverFlowEtaPhi", "OVER FLOW OCCUPANCY", ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

  rctTauVetoEtaPhi_ = ibooker.book2D("RctBitTauVetoEtaPhi", "TAU VETO OCCUPANCY", ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

  rctMipEtaPhi_ = ibooker.book2D("RctBitMipEtaPhi", "MIP OCCUPANCY", ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

  rctQuietEtaPhi_ = ibooker.book2D("RctBitQuietEtaPhi", "QUIET OCCUPANCY", ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

  rctHfPlusTauEtaPhi_ = ibooker.book2D("RctBitHfPlusTauEtaPhi", "HF plus Tau OCCUPANCY", ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    // rank histos
  rctRegionRank_ = ibooker.book1D("RctRegionRank", "REGION RANK", R10BINS, R10MIN, R10MAX);
  rctIsoEmRank_ = ibooker.book1D("RctEmIsoEmRank", "ISO EM RANK", R6BINS, R6MIN, R6MAX);
  rctNonIsoEmRank_ = ibooker.book1D("RctEmNonIsoEmRank", "NON-ISO EM RANK", R6BINS, R6MIN, R6MAX);
    // hw coordinates
//    rctEmCardRegion_ = dbe->book1D("rctEmCardRegion", "Em Card * Region",
//				   256, -127.5, 127.5);

    // bx histos
  rctRegionBx_ = ibooker.book1D("RctRegionBx", "Region BX", 256, -0.5, 4095.5);
  rctEmBx_ = ibooker.book1D("RctEmBx", "EM BX", 256, -0.5, 4095.5);
 
  //}
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

  e.getByToken(rctSource_L1CRCollection_,rgn);
 
  if (!rgn.isValid()) {
    edm::LogInfo("DataNotFound") << "can't find L1CaloRegionCollection";
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

  
  e.getByToken(rctSource_L1CEMCollection_,em);
  
  if (!em.isValid()) {
    edm::LogInfo("DataNotFound") << "can't find L1CaloEmCollection";
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
