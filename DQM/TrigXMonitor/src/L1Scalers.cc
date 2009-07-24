// $Id: L1Scalers.cc,v 1.14 2009/03/25 10:34:13 lorenzo Exp $
#include <iostream>


// FW
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ServiceRegistry/interface/Service.h"


// L1
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

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
  folderName_( ps.getUntrackedParameter< std::string>("dqmFolder", 
					  std::string("L1T/L1Scalers_EvF"))),
  l1scalers_(0),
  l1techScalers_(0),
  l1Correlations_(0),
  bxNum_(0),
  l1scalersBx_(0),
  l1techScalersBx_(0),
  nLumiBlock_(0),
  l1AlgoCounter_(0),
  l1TtCounter_(0),
  pixFedSize_(0),
  hfEnergy_(0),
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




void L1Scalers::beginJob(const edm::EventSetup& iSetup)
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
    l1techScalers_ = dbe_->book1D("l1TechAlgoBits", "L1 Tech. Trigger Bits",
				  64, -0.5, 63.5);
    l1techScalersBx_ = dbe_->book2D("l1TechAlgoBits_Vs_Bx", "L1 Technical "
				    "Trigger "
				    "Bits vs Bunch Number",
				    3600, -0.5, 3599.5, 64, -0.5, 63.5);
    bxNum_ = dbe_->book1D("bxNum", "Bunch number from GTFE",
			  3600, -0.5, 3599.5);

    nLumiBlock_ = dbe_->bookInt("nLumiBlock");


//  l1 total rate
    
    l1AlgoCounter_ = dbe_->bookInt("l1AlgoCounter");
    l1TtCounter_ = dbe_->bookInt("l1TtCounter");

    // early triggers
    pixFedSize_ = dbe_->book1D("pixFedSize", "Size of Pixel FED data",
			       200, 0., 20000.);
    hfEnergy_   = dbe_->book1D("hfEnergy", "HF energy",
			       100, 0., 500.);

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

    // First, the default
    // vector of bool
    DecisionWord gtDecisionWord = gtRecord->decisionWord();
    if ( ! gtDecisionWord.empty() ) { // if board not there this is zero
       // loop over dec. bit to get total rate (no overlap)
       for ( int i = 0; i < 128; ++i ) {
         if ( gtDecisionWord[i] ) {
	   rateAlgoCounter_++;
           l1AlgoCounter_->Fill(rateAlgoCounter_);
	   break;
	 }
       }
      // loop over decision bits
      for ( int i = 0; i < 128; ++i ) {
	if ( gtDecisionWord[i] ) {
	  l1scalers_->Fill(i);
	  l1scalersBx_->Fill(gtfeBx,i);
	  for ( int j = i + 1; j < 128; ++j ) {
	    if ( gtDecisionWord[j] ) {
	      l1Correlations_->Fill(i,j);
	      l1Correlations_->Fill(j,i);
	    }
	  }
	}
      }
    }   
    // loop over technical triggers
    // vector of bool again. 
    TechnicalTriggerWord tw = gtRecord->technicalTriggerWord();
    if ( ! tw.empty() ) {
       // loop over dec. bit to get total rate (no overlap)
       for ( int i = 0; i < 64; ++i ) {
         if ( tw[i] ) {
	   rateTtCounter_++;
           l1TtCounter_->Fill(rateTtCounter_);
	   break;
	 }
       }	 
      for ( int i = 0; i < 64; ++i ) {
	if ( tw[i] ) {
	  l1techScalers_->Fill(i);
	  l1techScalersBx_->Fill(gtfeBx, i);
	}
      } 
    } // ! tw.empty
  } // getbylabel succeeded


  // HACK
  // getting very basic uncalRH
  edm::Handle<FEDRawDataCollection> theRaw;
  bool getFed = e.getByLabel(fedRawCollection_, theRaw);
  if ( ! getFed ) {
    edm::LogInfo("FEDSizeFilter") << fedRawCollection_ << " not available";
  }
  else { // got the fed raw data
    unsigned int totalFEDsize = 0 ; 
    for (unsigned int i=fedStart_; i<=fedStop_; ++i) {
      LogDebug("Parameter") << "Examining fed " << i << " with size "
			    << theRaw->FEDData(i).size() ;
      totalFEDsize += theRaw->FEDData(i).size() ; 
    }
    pixFedSize_->Fill(totalFEDsize);
    
    LogDebug("Parameter") << "Total FED size: " << totalFEDsize;
  }      

  // HF - stolen from HLTrigger/special
  // getting very basic uncalRH
  edm::Handle<HFRecHitCollection> crudeHits;
  bool getHF = e.getByLabel(HcalRecHitCollection_, crudeHits);
  if ( ! getHF ) {
    LogDebug("Status") << HcalRecHitCollection_ << " not available";
  }
  else {

    LogDebug("Status") << "Filtering, with " << crudeHits->size() 
		       << " recHits to consider" ;
    for ( HFRecHitCollection::const_iterator hitItr = crudeHits->begin(); 
	  hitItr != crudeHits->end(); ++hitItr ) {     
      HFRecHit hit = (*hitItr);
     
      // masking noisy channels
      std::vector<int>::iterator result;
      result = std::find( maskedList_.begin(), maskedList_.end(), 
			  HcalDetId(hit.id()).hashed_index() );    
      if  (result != maskedList_.end()) 
	continue; 
      hfEnergy_->Fill(hit.energy());
       
    }
  }

  // END HACK

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


