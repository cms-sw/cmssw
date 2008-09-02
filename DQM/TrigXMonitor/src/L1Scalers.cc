// $Id$
#include <iostream>


// FW
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ServiceRegistry/interface/Service.h"


// L1
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"



#include "DataFormats/Scalers/interface/L1TriggerScalers.h"
#include "DataFormats/Scalers/interface/L1TriggerRates.h"
#include "DataFormats/Scalers/interface/LumiScalers.h"
#include "DQM/TrigXMonitor/interface/L1Scalers.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DQMServices/Core/interface/DQMStore.h"

using namespace edm;



L1Scalers::L1Scalers(const edm::ParameterSet &ps):
  dbe_(0), nev_(0),
  scalersSource_( ps.getParameter< edm::InputTag >("scalersResults")),
  verbose_(ps.getUntrackedParameter < bool > ("verbose", false)),
  l1GtDataSource_( ps.getParameter< edm::InputTag >("l1GtData"))
{
  LogDebug("Status") << "constructor" ;

  
} 




void L1Scalers::beginJob(const edm::EventSetup& iSetup)
{
  LogDebug("Status") << "L1Scalers::beginJob()...";

  dbe_ = Service<DQMStore>().operator->();
  if (dbe_ ) {
    dbe_->setVerbose(0);
    dbe_->setCurrentFolder("L1T/L1Scalers");

    if ( verbose_ ) {
      dbe_->setVerbose(1);
    }
    dbe_->setCurrentFolder("L1T/L1Scalers");

    l1scalers_ = dbe_->book1D("l1Scalers", "L1 scalers",
			      128, -0.5, 127.5);
    l1scalersBx_ = dbe_->book2D("l1ScalersBx", "L1 scalers vs Bunch Number",
				3600, -0.5, 3599.5,
				128, -0.5, 127.5);
    l1Correlations_ = dbe_->book2D("l1Correlations", "L1 scaler correlations"
				   " (locally derived)", 
				   128, -0.5, 127.5,
				   128, -0.5, 127.5);
    l1techScalers_ = dbe_->book1D("l1TechScalers", "L1 Technical Trigger "
				  "scalers ",
				  64, -0.5, 63.5);
    l1techScalersBx_ = dbe_->book2D("l1TechScalersBx", "L1 Technical Trigger "
				    "scalers vs Bunch Number",
				    3600, -0.5, 3599.5, 64, -0.5, 63.5);
    bxNum_ = dbe_->book1D("bxNum", "Bunch number from GTFE",
			  3600, -0.5, 3599.5);

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
      for ( int i = 0; i < 64; ++i ) {
	if ( tw[i] ) {
	  l1techScalers_->Fill(i);
	  l1techScalersBx_->Fill(gtfeBx, i);
	}
      } 
    } // ! tw.empty




    // now, the non-default BX numbers
    for ( int bx = 0; bx < 3600; ++bx ) {
      DecisionWord gtDecisionWord = gtRecord->decisionWord(bx);
      TechnicalTriggerWord tw = gtRecord->technicalTriggerWord(bx);
      
    }
  }
    

  return;
 
}

void L1Scalers::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
				    const edm::EventSetup& iSetup)
{

}


/// BeginRun
void L1Scalers::beginRun(const edm::Run& run, const edm::EventSetup& iSetup)
{
}

/// EndRun
void L1Scalers::endRun(const edm::Run& run, const edm::EventSetup& iSetup)
{
}


