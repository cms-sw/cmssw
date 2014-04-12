#include <iostream>


// FW
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

// L1
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTExtendedCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"


#include "FWCore/Framework/interface/LuminosityBlock.h"


#include "DataFormats/Scalers/interface/L1TriggerScalers.h"
#include "DataFormats/Scalers/interface/L1TriggerRates.h"
#include "DataFormats/Scalers/interface/LumiScalers.h"
#include "DQM/TrigXMonitor/interface/L1Scalers.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DQMServices/Core/interface/DQMStore.h"

// HACK START
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
// HACK END

using namespace edm;



L1Scalers::L1Scalers(const edm::ParameterSet &ps):
  dbe_(0), nev_(0),
  verbose_(ps.getUntrackedParameter < bool > ("verbose", false)),
  l1GtDataSource_( ps.getParameter< edm::InputTag >("l1GtData")),
  denomIsTech_(ps.getUntrackedParameter <bool> ("denomIsTech", true)),
  denomBit_(ps.getUntrackedParameter <unsigned int> ("denomBit", 40)),
  tfIsTech_(ps.getUntrackedParameter <bool> ("tfIsTech", true)),
  tfBit_(ps.getUntrackedParameter <unsigned int> ("tfBit", 41)),
  algoSelected_(ps.getUntrackedParameter<std::vector<unsigned int> >("algoMonitorBits", std::vector<unsigned int>())),
  techSelected_(ps.getUntrackedParameter<std::vector<unsigned int> >("techMonitorBits", std::vector<unsigned int>())),
  folderName_( ps.getUntrackedParameter< std::string>("dqmFolder", 
					  std::string("L1T/L1Scalers_EvF"))),
  l1scalers_(0),
  l1techScalers_(0),
  l1Correlations_(0),
  bxNum_(0),
  l1scalersBx_(0),
  l1techScalersBx_(0),
//   pixFedSizeBx_(0),
//   hfEnergyMaxTowerBx_(0),
  nLumiBlock_(0),
  l1AlgoCounter_(0),
  l1TtCounter_(0),
//   pixFedSize_(0),
//   hfEnergy_(0),
  fedStart_(ps.getUntrackedParameter<unsigned int>("firstFED", 0)),
  fedStop_(ps.getUntrackedParameter<unsigned int>("lastFED", 931)), 
  rateAlgoCounter_(0),					  
  rateTtCounter_(0),					  
  fedRawCollection_(ps.getParameter<edm::InputTag>("fedRawData")),
  maskedList_(ps.getUntrackedParameter<std::vector<int> >("maskedChannels", std::vector<int>())), //this is using the ashed index
  HcalRecHitCollection_(ps.getParameter<edm::InputTag>("HFRecHitCollection"))
{
  LogDebug("Status") << "constructor" ;
} 




void L1Scalers::beginJob(void)
{
  LogDebug("Status") << "L1Scalers::beginJob()...";

  dbe_ = Service<DQMStore>().operator->();
  if (dbe_ ) {
    dbe_->setVerbose(0);
    dbe_->setCurrentFolder(folderName_);


    l1scalers_ = dbe_->book1D("l1AlgoBits", "L1 Algorithm Bits",
			      128, -0.5, 127.5);
    l1scalersBx_ = dbe_->book2D("l1AlgoBits_Vs_Bx", "L1 Algorithm Bits vs "
				"Bunch Number",
				3600, -0.5, 3599.5,
				128, -0.5, 127.5);
    l1Correlations_ = dbe_->book2D("l1Correlations", "L1 Algorithm Bits " 
                                    "Correlations",
				   128, -0.5, 127.5,
				   128, -0.5, 127.5);
    l1techScalers_ = dbe_->book1D("l1TechBits", "L1 Tech. Trigger Bits",
				  64, -0.5, 63.5);
    l1techScalersBx_ = dbe_->book2D("l1TechBits_Vs_Bx", "L1 Technical "
				    "Trigger "
				    "Bits vs Bunch Number",
				    3600, -0.5, 3599.5, 64, -0.5, 63.5);
//     pixFedSizeBx_ = dbe_->book2D("pixFedSize_Vs_Bx", "Size of Pixel FED data vs "
// 				"Bunch Number",
// 				3600, -0.5, 3599.5,
// 				200, 0., 20000.);
//     hfEnergyMaxTowerBx_ = dbe_->book2D("hfEnergyMaxTower_Vs_Bx", "HF Energy Max Tower vs "
// 				"Bunch Number",
// 				3600, -0.5, 3599.5,
// 				100, 0., 500.);
    bxNum_ = dbe_->book1D("bxNum", "Bunch number from GTFE",
			  3600, -0.5, 3599.5);

    nLumiBlock_ = dbe_->bookInt("nLumiBlock");

//  l1 total rate
    
    l1AlgoCounter_ = dbe_->bookInt("l1AlgoCounter");
    l1TtCounter_ = dbe_->bookInt("l1TtCounter");


    //int maxNbins = 200;
    //int maxLumi = 2000;

    //timing plots
    std::stringstream sdenom;
    if(denomIsTech_) sdenom << "tech" ;
    else sdenom << "algo" ;

    dbe_->setCurrentFolder(folderName_ + "/Synch");
    algoBxDiff_.clear();
    algoBxDiff_.clear();
    algoBxDiffLumi_.clear();
    techBxDiffLumi_.clear();
    for(uint ibit = 0; ibit < algoSelected_.size(); ibit++){
      std::stringstream ss;
      ss << algoSelected_[ibit] << "_" << sdenom.str() << denomBit_;
      algoBxDiff_.push_back(dbe_->book1D("BX_diff_algo"+ ss.str(),"BX_diff_algo"+ ss.str(),9,-4,5));
      algoBxDiffLumi_.push_back(dbe_->book2D("BX_diffvslumi_algo"+ ss.str(),"BX_diff_algo"+ss.str(),MAX_LUMI_BIN,-0.5,double(MAX_LUMI_SEG)-0.5,9,-4,5));
      //algoBxDiffLumi_[ibit]->setAxisTitle("Lumi Section", 1);
    }
    for(uint ibit = 0; ibit < techSelected_.size(); ibit++){
      std::stringstream ss;
      ss << techSelected_[ibit] << "_" << sdenom.str() << denomBit_;
      techBxDiff_.push_back(dbe_->book1D("BX_diff_tech"+ ss.str(),"BX_diff_tech"+ ss.str(),9,-4,5));
      techBxDiffLumi_.push_back(dbe_->book2D("BX_diffvslumi_tech"+ ss.str(),"BX_diff_tech"+ss.str(),MAX_LUMI_BIN,-0.5,double(MAX_LUMI_SEG)-0.5,9,-4,5));
      //techBxDiffLumi_[ibit]->setAxisTitle("Lumi Section", 1);
    }

    //GMT timing plots
    std::stringstream ss1;
    ss1 << "_" << sdenom.str() << denomBit_;
    dtBxDiff_ = dbe_->book1D("BX_diff_DT" + ss1.str(),"BX_diff_DT" + ss1.str(),9,-4,5);
    dtBxDiffLumi_ = dbe_->book2D("BX_diffvslumi_DT" + ss1.str(),"BX_diffvslumi_DT" + ss1.str(),MAX_LUMI_BIN,-0.5,double(MAX_LUMI_SEG)-0.5,9,-4,5);
    cscBxDiff_ = dbe_->book1D("BX_diff_CSC" + ss1.str(),"BX_diff_CSC" + ss1.str(),9,-4,5);
    cscBxDiffLumi_ = dbe_->book2D("BX_diffvslumi_CSC" + ss1.str(),"BX_diffvslumi_CSC" + ss1.str(),MAX_LUMI_BIN,-0.5,double(MAX_LUMI_SEG)-0.5,9,-4,5);
    rpcbBxDiff_ = dbe_->book1D("BX_diff_RPCb" + ss1.str(),"BX_diff_RPCb" + ss1.str(),9,-4,5);
    rpcbBxDiffLumi_ = dbe_->book2D("BX_diffvslumi_RPCb" + ss1.str(),"BX_diffvslumi_RPCb" + ss1.str(),MAX_LUMI_BIN,-0.5,double(MAX_LUMI_SEG)-0.5,9,-4,5);
    rpcfBxDiff_ = dbe_->book1D("BX_diff_RPCf" + ss1.str(),"BX_diff_RPCf" + ss1.str(),9,-4,5);
    rpcfBxDiffLumi_ = dbe_->book2D("BX_diffvslumi_RPCf" + ss1.str(),"BX_diffvslumi_RPCf" + ss1.str(),MAX_LUMI_BIN,-0.5,double(MAX_LUMI_SEG)-0.5,9,-4,5);


    // early triggers
//     pixFedSize_ = dbe_->book1D("pixFedSize", "Size of Pixel FED data",
// 			       200, 0., 20000.);
//     hfEnergy_   = dbe_->book1D("hfEnergy", "HF energy",
// 			       100, 0., 500.);

  }
  
  
  return;
}

void L1Scalers::endJob(void)
{
}

void L1Scalers::analyze(const edm::Event &e, const edm::EventSetup &iSetup)
{
  nev_++;

  LogDebug("Status") << "L1Scalers::analyze  event " << nev_ ;

  // int myGTFEbx = -1;
  // get Global Trigger decision and the decision word
  // these are locally derived
  edm::Handle<L1GlobalTriggerReadoutRecord> gtRecord;
  bool t = e.getByLabel(l1GtDataSource_,gtRecord);

  if ( ! t ) {
    LogDebug("Product") << "can't find L1GlobalTriggerReadoutRecord "
			<< "with label " << l1GtDataSource_.label() ;
  }
  else {

    // DEBUG
    //gtRecord->print(std::cout);
    // DEBUG
    
    L1GtfeWord gtfeWord = gtRecord->gtfeWord();
    int gtfeBx = gtfeWord.bxNr();
    bxNum_->Fill(gtfeBx);
    // myGTFEbx = gtfeBx;
    
    bool tfBitGood = false;

    // First, the default
    // vector of bool
    for(int iebx=0; iebx<=4; iebx++) {

      //Algorithm Bits
      DecisionWord gtDecisionWord = gtRecord->decisionWord(iebx-2);
      //    DecisionWord gtDecisionWord = gtRecord->decisionWord();
      if ( ! gtDecisionWord.empty() ) { // if board not there this is zero
	// loop over dec. bit to get total rate (no overlap)
	for ( uint i = 0; i < gtDecisionWord.size(); ++i ) {
	  if ( gtDecisionWord[i] ) {
	    rateAlgoCounter_++;
	    l1AlgoCounter_->Fill(rateAlgoCounter_);
	    break;
	  }
	}
	// loop over decision bits
	for ( uint i = 0; i < gtDecisionWord.size(); ++i ) {
	  if ( gtDecisionWord[i] ) {
	    l1scalers_->Fill(i);
	    l1scalersBx_->Fill(gtfeBx-2+iebx,i);
	    for ( uint j = i + 1; j < gtDecisionWord.size(); ++j ) {
	      if ( gtDecisionWord[j] ) {
		l1Correlations_->Fill(i,j);
		l1Correlations_->Fill(j,i);
	      }
	    }
	  }
	}
      }//!empty DecisionWord
 

      // loop over technical triggers
      // vector of bool again. 
      TechnicalTriggerWord tw = gtRecord->technicalTriggerWord(iebx-2);
      //    TechnicalTriggerWord tw = gtRecord->technicalTriggerWord();
      if ( ! tw.empty() ) {
	// loop over dec. bit to get total rate (no overlap)
	for ( uint i = 0; i < tw.size(); ++i ) {
	  if ( tw[i] ) {
	    rateTtCounter_++;
	    l1TtCounter_->Fill(rateTtCounter_);
	    break;
	  }
	}	 
	for ( uint i = 0; i < tw.size(); ++i ) {
	  if ( tw[i] ) {
	    l1techScalers_->Fill(i);
	    l1techScalersBx_->Fill(gtfeBx-2+iebx, i);
	  }
	} 

	// check if bit used to filter timing plots fired in this event 
	// (anywhere in the bx window)
	if(tfIsTech_){
	  if(tfBit_ < tw.size()){
	    if( tw[tfBit_] ) tfBitGood = true;
	  }
	}
      } // ! tw.empty

    }//bx


    //timing plots
    earliestDenom_ = 9;
    earliestAlgo_.clear();
    earliestTech_.clear();
    for(uint i=0; i < techSelected_.size(); i++) earliestTech_.push_back(9);
    for(uint i=0; i < algoSelected_.size(); i++) earliestAlgo_.push_back(9);

    //GMT information
    edm::Handle<L1MuGMTReadoutCollection> gmtCollection;
    e.getByLabel(l1GtDataSource_,gmtCollection);
    

    if (!gmtCollection.isValid()) {
      edm::LogInfo("DataNotFound") << "can't find L1MuGMTReadoutCollection with label "
				   << l1GtDataSource_.label() ;
    }

    // remember the bx of 1st candidate of each system (9=none)
    int bx1st[4] = {9, 9, 9, 9};
      
    if(tfBitGood){//to avoid single BSC hits

      for(int iebx=0; iebx<=4; iebx++) {
	TechnicalTriggerWord tw = gtRecord->technicalTriggerWord(iebx-2);
	DecisionWord gtDecisionWord = gtRecord->decisionWord(iebx-2);

	bool denomBitGood = false;

	//check if reference bit is valid
	if(denomIsTech_){
	  if ( ! tw.empty() ) {
	    if( denomBit_ < tw.size() ){
	      denomBitGood = true;
	      if( tw[denomBit_] && earliestDenom_==9 ) earliestDenom_ = iebx; 
	    }
	  }
	}
	else{
	  if ( ! gtDecisionWord.empty() ) { 
	    if( denomBit_ < gtDecisionWord.size() ){
	      denomBitGood = true;
	      if( gtDecisionWord[denomBit_] && earliestDenom_==9 ) earliestDenom_ = iebx; 
	    }
	  }
	}

	if(denomBitGood){

	  //get earliest tech bx's
	  if ( ! tw.empty() ) {
	    for(uint ibit = 0; ibit < techSelected_.size(); ibit++){	  
	      if(techSelected_[ibit] < tw.size()){
		if(tw[techSelected_[ibit]] && earliestTech_[ibit] == 9) earliestTech_[ibit] = iebx;
	      }
	    }
	  }

	  //get earliest algo bx's
	  if(!gtDecisionWord.empty()){	    
	    for(uint ibit = 0; ibit < algoSelected_.size(); ibit++){	  
	      if(algoSelected_[ibit] < gtDecisionWord.size()){
		if(gtDecisionWord[algoSelected_[ibit]] && earliestAlgo_[ibit] == 9) earliestAlgo_[ibit] = iebx;
	      }
	    }
	  }

	}//denomBitGood

      }//bx


      //get earliest single muon trigger system bx's
      if (gmtCollection.isValid()) {

	// get GMT readout collection
	L1MuGMTReadoutCollection const* gmtrc = gmtCollection.product();
	// get record vector
	std::vector<L1MuGMTReadoutRecord> gmt_records = gmtrc->getRecords();
	// loop over records of individual bx's
	std::vector<L1MuGMTReadoutRecord>::const_iterator RRItr;

	for( RRItr = gmt_records.begin(); RRItr != gmt_records.end(); RRItr++ ) {//loop from BX=-2 to BX=2
	  std::vector<L1MuRegionalCand> INPCands[4] = {
	    RRItr->getDTBXCands(),
	    RRItr->getBrlRPCCands(),
	    RRItr->getCSCCands(),
	    RRItr->getFwdRPCCands()
	  };
	  std::vector<L1MuRegionalCand>::const_iterator INPItr;
	  int BxInEvent = RRItr->getBxInEvent();
	  
	  // find the first non-empty candidate in this bx
	  for(int i=0; i<4; i++) {//for each single muon trigger system
	    for( INPItr = INPCands[i].begin(); INPItr != INPCands[i].end(); ++INPItr ) {
	      if(!INPItr->empty()) {
		if(bx1st[i]==9) bx1st[i]=BxInEvent+2;//must go from 0 to 4 (consistent with above)
	      }
	    }      
	  }
	  //for(int i=0; i<4; i++) 
	  //	std::cout << "bx1st[" << i << "] = " << bx1st[i];
	  //std::cout << std::endl;
	}

      }//gmtCollection.isValid


      //calculated bx difference
      if(earliestDenom_ != 9){
	for(uint ibit = 0; ibit < techSelected_.size(); ibit++){	  
	  if(earliestTech_[ibit] != 9){
	    int diff = earliestTech_[ibit] - earliestDenom_ ;
	    techBxDiff_[ibit]->Fill(diff);
	    techBxDiffLumi_[ibit]->Fill(e.luminosityBlock(), diff);
	  }
	}
	for(uint ibit = 0; ibit < algoSelected_.size(); ibit++){	  
	  if(earliestAlgo_[ibit] != 9){
	    int diff = earliestAlgo_[ibit] - earliestDenom_ ;
	    algoBxDiff_[ibit]->Fill(diff);
	    algoBxDiffLumi_[ibit]->Fill(e.luminosityBlock(), diff);
	  }
	}

	if(bx1st[0] != 9){
	  int diff = bx1st[0] - earliestDenom_;
	  dtBxDiff_->Fill(diff);
	  dtBxDiffLumi_->Fill(e.luminosityBlock(), diff);
	}
	if(bx1st[1] != 9){
	  int diff = bx1st[1] - earliestDenom_;
	  rpcbBxDiff_->Fill(diff);
	  rpcbBxDiffLumi_->Fill(e.luminosityBlock(), diff);
	}
	if(bx1st[2] != 9){
	  int diff = bx1st[2] - earliestDenom_;
	  cscBxDiff_->Fill(diff);
	  cscBxDiffLumi_->Fill(e.luminosityBlock(), diff);
	}
	if(bx1st[3] != 9){
	  int diff = bx1st[3] - earliestDenom_;
	  rpcfBxDiff_->Fill(diff);
	  rpcfBxDiffLumi_->Fill(e.luminosityBlock(), diff);
	}

      }

    }//tt41Good

  }
  
  return;
 
}

void L1Scalers::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
				    const edm::EventSetup& iSetup)
{
  nLumiBlock_->Fill(lumiSeg.id().luminosityBlock());

}


/// BeginRun
void L1Scalers::beginRun(const edm::Run& run, const edm::EventSetup& iSetup)
{
}

/// EndRun
void L1Scalers::endRun(const edm::Run& run, const edm::EventSetup& iSetup)
{
}


