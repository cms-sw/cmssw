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
   histFolder_ (ps.getUntrackedParameter<std::string>("HistFolder", "L1T/L1TRCT")),
   rctSource_L1CRCollection_( consumes<L1CaloRegionCollection>(ps.getParameter< InputTag >("rctSource") )),
   rctSource_L1CEMCollection_( consumes<L1CaloEmCollection>(ps.getParameter< InputTag >("rctSource") )),
   rctSource_GCT_L1CRCollection_( consumes<L1CaloRegionCollection>(ps.getParameter< InputTag >("gctSource") )),
   rctSource_GCT_L1CEMCollection_( consumes<L1CaloEmCollection>(ps.getParameter< InputTag >("gctSource") )),
   filterTriggerType_ (ps.getParameter< int >("filterTriggerType")),
   selectBX_ (ps.getUntrackedParameter< int >("selectBX",2))
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
}

void L1TRCT::beginLuminosityBlock(const edm::LuminosityBlock& l, const edm::EventSetup& c){
}


void L1TRCT::bookHistograms(DQMStore::IBooker &ibooker, edm::Run const&, edm::EventSetup const&)
{
  nev_ = 0;
  ibooker.setCurrentFolder(histFolder_);

  triggerType_ = ibooker.book1D("TriggerType", "TriggerType", 17, -0.5, 16.5);

  // RCT UNPACKER
  // electrons
  rctIsoEmEtEtaPhi_ =	ibooker.book2D("RctEmIsoEmEtEtaPhi", "ISO EM E_{T}", ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);
  rctIsoEmOccEtaPhi_ = ibooker.book2D("RctEmIsoEmOccEtaPhi", "ISO EM OCCUPANCY", ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);
  rctNonIsoEmEtEtaPhi_ = ibooker.book2D("RctEmNonIsoEmEtEtaPhi", "NON-ISO EM E_{T}", ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);
  rctNonIsoEmOccEtaPhi_ = ibooker.book2D("RctEmNonIsoEmOccEtaPhi", "NON-ISO EM OCCUPANCY",ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

  // global regions
  rctRegionsEtEtaPhi_ = ibooker.book2D("RctRegionsEtEtaPhi", "REGION E_{T}", ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);
  rctRegionsOccEtaPhi_ = ibooker.book2D("RctRegionsOccEtaPhi", "REGION OCCUPANCY", ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

  // bits
  rctOverFlowEtaPhi_ = ibooker.book2D("RctBitOverFlowEtaPhi", "OVER FLOW OCCUPANCY", ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);
  rctTauVetoEtaPhi_ = ibooker.book2D("RctBitTauVetoEtaPhi", "TAU VETO OCCUPANCY", ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);
  rctMipEtaPhi_ = ibooker.book2D("RctBitMipEtaPhi", "MIP OCCUPANCY", ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);
  rctQuietEtaPhi_ = ibooker.book2D("RctBitQuietEtaPhi", "QUIET OCCUPANCY", ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);
  rctHfPlusTauEtaPhi_ = ibooker.book2D("RctBitHfPlusTauEtaPhi", "HF plus Tau OCCUPANCY", ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

  // rank histos
  rctRegionRank_ = ibooker.book1D("RctRegionRank", "REGION RANK", R10BINS, R10MIN, R10MAX);
  rctIsoEmRank_ = ibooker.book1D("RctEmIsoEmRank", "ISO EM RANK", R6BINS, R6MIN, R6MAX);
  rctNonIsoEmRank_ = ibooker.book1D("RctEmNonIsoEmRank", "NON-ISO EM RANK", R6BINS, R6MIN, R6MAX);

  // bx histos
  rctRegionBx_ = ibooker.book1D("RctRegionBx", "Region BX", 10, -2.5, 7.5);
  rctEmBx_ = ibooker.book1D("RctEmBx", "EM BX", 10, -2.5, 7.5); 

  // NOT CENTRAL BXs
  rctNotCentralRegionsEtEtaPhi_ = ibooker.book2D("rctNotCentralRegionsEtEtaPhi", "REGION E_{T}", ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);
  rctNotCentralRegionsOccEtaPhi_ = ibooker.book2D("rctNotCentralRegionsOccEtaPhi", "REGION OCCUPANCY", ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);
  rctNotCentralIsoEmEtEtaPhi_ =   ibooker.book2D("rctNotCentralEmIsoEmEtEtaPhi", "ISO EM E_{T}", ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);
  rctNotCentralIsoEmOccEtaPhi_ = ibooker.book2D("rctNotCentralEmIsoEmOccEtaPhi", "ISO EM OCCUPANCY", ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);
  rctNotCentralNonIsoEmEtEtaPhi_ = ibooker.book2D("rctNotCentralEmNonIsoEmEtEtaPhi", "NON-ISO EM E_{T}", ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);
  rctNotCentralNonIsoEmOccEtaPhi_ = ibooker.book2D("rctNotCentralEmNonIsoEmOccEtaPhi", "NON-ISO EM OCCUPANCY",ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);


  // GCT UNPACKER

  // electrons
  layer2IsoEmEtEtaPhi_ =   ibooker.book2D("Layer2EmIsoEmEtEtaPhi", "ISO EM E_{T}", ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);
  layer2IsoEmOccEtaPhi_ = ibooker.book2D("Layer2EmIsoEmOccEtaPhi", "ISO EM OCCUPANCY", ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);
  layer2NonIsoEmEtEtaPhi_ = ibooker.book2D("Layer2EmNonIsoEmEtEtaPhi", "NON-ISO EM E_{T}", ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);
  layer2NonIsoEmOccEtaPhi_ = ibooker.book2D("Layer2EmNonIsoEmOccEtaPhi", "NON-ISO EM OCCUPANCY",ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

  // global regions
  layer2RegionsEtEtaPhi_ = ibooker.book2D("Layer2RegionsEtEtaPhi", "REGION E_{T}", ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);
  layer2RegionsOccEtaPhi_ = ibooker.book2D("Layer2RegionsOccEtaPhi", "REGION OCCUPANCY", ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

  // bits
  layer2OverFlowEtaPhi_ = ibooker.book2D("Layer2BitOverFlowEtaPhi", "OVER FLOW OCCUPANCY", ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);
  layer2TauVetoEtaPhi_ = ibooker.book2D("Layer2BitTauVetoEtaPhi", "TAU VETO OCCUPANCY", ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);
  layer2MipEtaPhi_ = ibooker.book2D("Layer2BitMipEtaPhi", "MIP OCCUPANCY", ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);
  layer2QuietEtaPhi_ = ibooker.book2D("Layer2BitQuietEtaPhi", "QUIET OCCUPANCY", ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);
  layer2HfPlusTauEtaPhi_ = ibooker.book2D("Layer2BitHfPlusTauEtaPhi", "HF plus Tau OCCUPANCY", ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

  // rank histos
  layer2RegionRank_ = ibooker.book1D("Layer2RegionRank", "REGION RANK", R10BINS, R10MIN, R10MAX);
  layer2IsoEmRank_ = ibooker.book1D("Layer2EmIsoEmRank", "ISO EM RANK", R6BINS, R6MIN, R6MAX);
  layer2NonIsoEmRank_ = ibooker.book1D("Layer2EmNonIsoEmRank", "NON-ISO EM RANK", R6BINS, R6MIN, R6MAX);

  // bx histos
  layer2RegionBx_ = ibooker.book1D("Layer2RegionBx", "Region BX", 10, -2.5, 7.5);
  layer2EmBx_ = ibooker.book1D("Layer2EmBx", "EM BX", 10, -2.5, 7.5); 

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
  e.getByToken(rctSource_L1CEMCollection_,em);

  // Get the Layer2 digis
  edm::Handle < L1CaloEmCollection > emLayer2;
  edm::Handle < L1CaloRegionCollection > rgnLayer2;

  bool doEmLayer2 = true;
  bool doHdLayer2 = true;

  e.getByToken(rctSource_GCT_L1CRCollection_,rgnLayer2);
  e.getByToken(rctSource_GCT_L1CEMCollection_,emLayer2);

 
  if (!rgn.isValid()) {
    edm::LogInfo("DataNotFound") << "can't find L1CaloRegionCollection - RCT";
    doHd = false;
  }

  if (!em.isValid()) {
    edm::LogInfo("DataNotFound") << "can't find L1CaloEmCollection - Layer2 ";
    doEm = false;
  }

  if (!rgnLayer2.isValid()) {
    edm::LogInfo("DataNotFound") << "can't find L1CaloRegionCollection - GCT";
    doHdLayer2 = false;
  }

  if (!emLayer2.isValid()) {
    edm::LogInfo("DataNotFound") << "can't find L1CaloEmCollection - Layer2";
    doEmLayer2 = false;
  }




  if ( doHd ) {
    // Fill the RCT histograms

    for (L1CaloRegionCollection::const_iterator ireg = rgn->begin();
       ireg != rgn->end(); ireg++) {

      if(ireg->et()>0)
      {
      rctRegionBx_->Fill(ireg->bx());
      } 

      if(selectBX_==-1 || selectBX_==ireg->bx()) {

      if(ireg->et()>0){

      rctRegionRank_->Fill(ireg->et());
      if(ireg->et()>5){
	rctRegionsOccEtaPhi_->Fill(ireg->gctEta(), ireg->gctPhi());
      }
      rctRegionsEtEtaPhi_->Fill(ireg->gctEta(), ireg->gctPhi(), ireg->et());
      }

    if(ireg->overFlow())  rctOverFlowEtaPhi_ ->Fill(ireg->gctEta(), ireg->gctPhi());
    if(ireg->tauVeto())   rctTauVetoEtaPhi_  ->Fill(ireg->gctEta(), ireg->gctPhi());
    if(ireg->mip())       rctMipEtaPhi_      ->Fill(ireg->gctEta(), ireg->gctPhi());
    if(ireg->quiet())     rctQuietEtaPhi_    ->Fill(ireg->gctEta(), ireg->gctPhi());
    if(ireg->fineGrain()) rctHfPlusTauEtaPhi_->Fill(ireg->gctEta(), ireg->gctPhi()); 
    
    }
    else if (selectBX_!=-1 && selectBX_!=ireg->bx()){
      if(ireg->et()>5) rctNotCentralRegionsOccEtaPhi_->Fill(ireg->gctEta(), ireg->gctPhi());
      rctNotCentralRegionsEtEtaPhi_->Fill(ireg->gctEta(), ireg->gctPhi(), ireg->et());
    } 


 }

 } 

  if (doEm){
  // Isolated and non-isolated EM
  for (L1CaloEmCollection::const_iterator iem = em->begin();
       iem != em->end(); iem++) {

      if(iem->rank()==0) continue;
      rctEmBx_->Fill(iem->bx());
      if(selectBX_==-1 || selectBX_==iem->bx()) {

    if (iem->isolated()) {
      rctIsoEmRank_->Fill(iem->rank());
      rctIsoEmEtEtaPhi_->Fill(iem->regionId().ieta(),
			      iem->regionId().iphi(), iem->rank());
      if(iem->rank()>10){
	rctIsoEmOccEtaPhi_->Fill(iem->regionId().ieta(),
				 iem->regionId().iphi());
      }
    }
    else {
      rctNonIsoEmRank_->Fill(iem->rank());
      rctNonIsoEmEtEtaPhi_->Fill(iem->regionId().ieta(),
				 iem->regionId().iphi(), iem->rank());
      if(iem->rank()>10){
	rctNonIsoEmOccEtaPhi_->Fill(iem->regionId().ieta(),
				    iem->regionId().iphi());
      }
    }
    }
    else if (selectBX_!=-1 && selectBX_!=iem->bx()) {
    if (iem->isolated()) {
      rctNotCentralIsoEmEtEtaPhi_->Fill(iem->regionId().ieta(),
                        iem->regionId().iphi(), iem->rank());
      if(iem->rank()>10){
      rctNotCentralIsoEmOccEtaPhi_->Fill(iem->regionId().ieta(),
                         iem->regionId().iphi());
      }
    }
    else {
      rctNotCentralNonIsoEmEtEtaPhi_->Fill(iem->regionId().ieta(),
                         iem->regionId().iphi(), iem->rank());
      if(iem->rank()>10){
      rctNotCentralNonIsoEmOccEtaPhi_->Fill(iem->regionId().ieta(),
                            iem->regionId().iphi());
      }
      }
    }
  }

}

  // Layer2 Histograms
  
  if ( doHdLayer2 ) {
    // Fill the RCT histograms

    for (L1CaloRegionCollection::const_iterator ireg = rgnLayer2->begin();
       ireg != rgnLayer2->end(); ireg++) {

      if(ireg->et()>0){

      layer2RegionBx_->Fill(ireg->bx());

      layer2RegionRank_->Fill(ireg->et());
      if(ireg->et()>5){
	layer2RegionsOccEtaPhi_->Fill(ireg->gctEta(), ireg->gctPhi());
      }
      layer2RegionsEtEtaPhi_->Fill(ireg->gctEta(), ireg->gctPhi(), ireg->et());
      }

    if(ireg->overFlow())  layer2OverFlowEtaPhi_ ->Fill(ireg->gctEta(), ireg->gctPhi());
    if(ireg->tauVeto())   layer2TauVetoEtaPhi_  ->Fill(ireg->gctEta(), ireg->gctPhi());
    if(ireg->mip())       layer2MipEtaPhi_      ->Fill(ireg->gctEta(), ireg->gctPhi());
    if(ireg->quiet())     layer2QuietEtaPhi_    ->Fill(ireg->gctEta(), ireg->gctPhi());
    if(ireg->fineGrain()) layer2HfPlusTauEtaPhi_->Fill(ireg->gctEta(), ireg->gctPhi()); 
    
    }

 } 

  if (doEmLayer2 ) {
  // Isolated and non-isolated EM
  for (L1CaloEmCollection::const_iterator iem = emLayer2->begin();
       iem != emLayer2->end(); iem++) {

      if(iem->rank()==0) continue;
      layer2EmBx_->Fill(iem->bx());

    if (iem->isolated()) {
      layer2IsoEmRank_->Fill(iem->rank());
      layer2IsoEmEtEtaPhi_->Fill(iem->regionId().ieta(),
			      iem->regionId().iphi(), iem->rank());
      if(iem->rank()>10){
	layer2IsoEmOccEtaPhi_->Fill(iem->regionId().ieta(),
				 iem->regionId().iphi());
      }
    }
    else {
      layer2NonIsoEmRank_->Fill(iem->rank());
      layer2NonIsoEmEtEtaPhi_->Fill(iem->regionId().ieta(),
				 iem->regionId().iphi(), iem->rank());
      if(iem->rank()>10){
	layer2NonIsoEmOccEtaPhi_->Fill(iem->regionId().ieta(),
				    iem->regionId().iphi());
      }
    }

  }

  }

}
