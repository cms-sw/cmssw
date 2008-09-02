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
  dbe_(0),
  scalersSource_( ps.getParameter< edm::InputTag >("scalersResults")),
  verbose_(ps.getUntrackedParameter < bool > ("verbose", false)),
  l1GtDataSource_( ps.getParameter< edm::InputTag >("l1GtData"))
{
  LogDebug("Status") << "constructor" ;

  dbe_ = Service<DQMStore>().operator->();
  if (dbe_ ) {
    dbe_->setVerbose(0);
    dbe_->setCurrentFolder("L1T/L1Scalers");
  
    //  bins and ranges to be fixed;

    dbe_->setCurrentFolder("L1T/L1Scalers/L1TriggerScalers");
    
    orbitNum = dbe_->book1D("Orbit Number","Orbit Number", 1000,0,1000);
    trigNum = dbe_->book1D("Trigger Number","trigger Number",1000,0,1000);
    eventNum = dbe_->book1D("event Number","event Number", 1000,0,1000);
    finalTrig = dbe_->book1D("Final Triggers","Final Triggers", 1000,0,1000);
    randTrig = dbe_->book1D("Random Triggers","Random Triggers", 1000,0,1000);
    numberResets = dbe_->book1D("Number Resets","Number Resets", 1000,0,1000);
    deadTime = dbe_->book1D("DeadTime","DeadTime", 1000,0,1000);
    lostFinalTriggers = dbe_->book1D("Lost Final Trigger","Lost Final Trigger",
				     1000,0,1000);
    
    dbe_->setCurrentFolder("L1T/L1Scalers/L1TriggerRates");
    orbitNumRate = dbe_->book1D("Orbit Number Rate","Orbit Number Rate", 
				1000,0,1000);
    trigNumRate = dbe_->book1D("Trigger Number Rate","trigger Number Rate",
			       1000,0,1000);
    eventNumRate = dbe_->book1D("event Number rate","event Number Rate", 
				1000,0,1000);
    finalTrigRate = dbe_->book1D("Final Trigger Rate","Final Trigger Rate", 
				 1000,0,1000);
    randTrigRate = dbe_->book1D("Random Trigger Rate","Random Trigger Rate", 
				1000,0,1000);
    numberResetsRate = dbe_->book1D("Number Resets Rate","Number Resets Rate",
				    1000,0,1000);
    deadTimePercent = dbe_->book1D("DeadTimepercent","DeadTimePercent", 
				   1000,0,1000);
    lostFinalTriggersPercent = dbe_->book1D("Lost Final Trigger Percent",
					    "Lost Final Triggerpercent", 
					    1000,0,1000);
    
    dbe_->setCurrentFolder("L1T/L1Scalers/LumiScalers");
    instLumi = dbe_->book1D("Instant Lumi","Instant Lumi",1000,0,1000);
    instLumiErr = dbe_->book1D("Instant Lumi Err","Instant Lumi Err",1000,
			       0,1000);
    instLumiQlty = dbe_->book1D("Instant Lumi Qlty","Instant Lumi Qlty",1000,
				0,1000);
    instEtLumi = dbe_->book1D("Instant Et Lumi","Instant Et Lumi",1000,0,1000);
    instEtLumiErr = dbe_->book1D("Instant Et Lumi Err","Instant Et Lumi Err",
				 1000,0,1000);
    instEtLumiQlty = dbe_->book1D("Instant Et Lumi Qlty",
				  "Instant Et Lumi Qlty",1000,0,1000);
    sectionNum = dbe_->book1D("Section Number","Section Number",1000,0,1000);
    startOrbit = dbe_->book1D("Start Orbit","Start Orbit",1000,0,1000);
    numOrbits = dbe_->book1D("Num Orbits","Num Orbits",1000,0,1000);
    
    
    nev_=0;
  }
  
} 




void L1Scalers::beginJob(const edm::EventSetup& iSetup)
{
  LogDebug("Status") << "L1Scalers::beginJob()...";

  if (dbe_) {
    if ( verbose_ ) {
      dbe_->setVerbose(1);
    }
    dbe_->setCurrentFolder("L1T/L1Scalers");
    // fixed - only for 128 algo bits right now
    l1scalers_ = dbe_->book1D("l1Scalers", "L1 scalers (locally derived)",
			      128, -0.5, 127.5);
    l1Correlations_ = dbe_->book2D("l1Correlations", "L1 scaler correlations"
				   " (locally derived)", 
				   128, -0.5, 127.5,
				   128, -0.5, 127.5);
    l1techScalers_ = dbe_->book1D("l1TechScalers", "L1 Technical Trigger "
				  "scalers (locally derived)",
				  64, -0.5, 63.5);

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
  edm::Handle<L1GlobalTriggerReadoutRecord> myGTReadoutRecord;
  bool t = e.getByLabel(l1GtDataSource_,myGTReadoutRecord);
  if ( ! t ) {
    edm::LogInfo("Product") << "can't find L1GlobalTriggerReadoutRecord "
			   << "with label " << l1GtDataSource_.label() ;
  }
  else {

    // DEBUG
    LogDebug("Status") << "dumping readout record" ;
    myGTReadoutRecord->print(std::cout);
    // DEBUG



    // vector of bool
    DecisionWord gtDecisionWord = myGTReadoutRecord->decisionWord();
    if ( ! gtDecisionWord.empty() ) { // if board not there this is zero
      // loop over decision bits
      for ( int i = 0; i < 128; ++i ) {
	if ( gtDecisionWord[i] ) {
	  l1scalers_->Fill(i);
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
    TechnicalTriggerWord tw = myGTReadoutRecord->technicalTriggerWord();
    if ( ! tw.empty() ) {
      for ( int i = 0; i < 64; ++i ) {
	if ( tw[i] ) {
	  l1techScalers_->Fill(i);
	}
      } 
    } // ! tw.empty
  }
    

  return;

  // SCAL data
  edm::Handle<L1TriggerScalersCollection> triggerScalers;
  bool a = e.getByLabel(scalersSource_, triggerScalers);
  edm::Handle<L1TriggerRatesCollection> triggerRates;
  bool b = e.getByLabel(scalersSource_, triggerRates);
  edm::Handle<LumiScalersCollection> lumiScalers;
  bool c = e.getByLabel(scalersSource_, lumiScalers);
  if ( ! (a && b && c ) ) {
    LogInfo("Status") << "getByLabel failed with label " 
		      << scalersSource_;
  }
  else { // we have the data 
  
    L1TriggerScalersCollection::const_iterator it = triggerScalers->begin();
  
    if(triggerScalers->size()){ 
      orbitNum ->Fill(it->orbitNumber());
      trigNum ->Fill(it->triggerNumber());
      eventNum ->Fill(it->eventNumber());
      finalTrig ->Fill(it->finalTriggersDistributed());
      randTrig ->Fill(it->randomTriggers());
      numberResets ->Fill(it->numberResets());
      deadTime ->Fill(it->deadTime());
      lostFinalTriggers ->Fill(it->lostFinalTriggers());
    }
  
    L1TriggerRatesCollection::const_iterator it2 = triggerRates->begin();
 
    if(triggerRates->size()){ 
      orbitNumRate ->Fill(it2->orbitNumberRate());
      trigNumRate ->Fill(it2->triggerNumberRate());
      eventNumRate ->Fill(it2->eventNumberRate());
      finalTrigRate ->Fill(it2->finalTriggersDistributedRate());
      randTrigRate ->Fill(it2->randomTriggersRate());
      numberResetsRate ->Fill(it2->numberResetsRate());
      deadTimePercent ->Fill(it2->deadTimePercent());
      lostFinalTriggersPercent ->Fill(it2->lostFinalTriggersPercent());
    }
 
    LumiScalersCollection::const_iterator it3 = lumiScalers->begin();
 
    if(lumiScalers->size()){ 
   
      instLumi->Fill(it3->instantLumi());
      instLumiErr->Fill(it3->instantLumiErr()); 
      instLumiQlty->Fill(it3->instantLumiQlty()); 
      instEtLumi->Fill(it3->instantETLumi()); 
      instEtLumiErr->Fill(it3->instantETLumiErr()); 
      instEtLumiQlty->Fill(it3->instantETLumiQlty()); 
      sectionNum->Fill(it3->sectionNumber()); 
      startOrbit->Fill(it3->startOrbit()); 
      numOrbits->Fill(it3->numOrbits()); 
  
    }
  } // getByLabel succeeds for scalers

 
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


