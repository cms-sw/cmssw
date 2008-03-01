#include <iostream>


// FW
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ServiceRegistry/interface/Service.h"


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
//  triggerScalersSource_( ps.getParameter< edm::InputTag >("TriggerScalersResults")),
//  triggerRatesSource_( ps.getParameter< edm::InputTag >("TriggerRatesResults")),
//  lumiScalersSource_( ps.getParameter< edm::InputTag >("LumiScalersResults")),
  verbose_(ps.getUntrackedParameter < bool > ("verbose", false)),
  monitorDaemon_(ps.getUntrackedParameter<bool>("MonitorDaemon", false))
{
  if ( verbose_ ) {
    std::cout << "L1Scalers::L1Scalers(ParameterSet) called...." 
	      << std::endl;
  }

  if (verbose_)
    std::cout << "L1Scalers: constructor...." << std::endl;

  dbe_ = Service<DQMStore>().operator->();
  dbe_->setVerbose(0);

  outputFile_ =
      ps.getUntrackedParameter < std::string > ("outputFile", "");
  if (outputFile_.size() != 0) {
    std::cout << "L1T Monitoring histograms will be saved to " 
	      << outputFile_ << std::endl;
  }
  else {
    outputFile_ = "ScalersDQM.root";
  }

  bool disable =
      ps.getUntrackedParameter < bool > ("disableROOToutput", false);
  if (disable) {
    outputFile_ = "";
  }


  if (dbe_ ) {
    dbe_->setCurrentFolder("L1T/L1Scalers");
  }
  
//  bins and ranges to be fixed;

 dbe_->setCurrentFolder("L1T/L1Scalers/L1TriggerScalers");
  
 orbitNum = dbe_->book1D("Orbit Number","Orbit Number", 1000,0,1000);
 trigNum = dbe_->book1D("Trigger Number","trigger Number",1000,0,1000);
 eventNum = dbe_->book1D("event Number","event Number", 1000,0,1000);
 finalTrig = dbe_->book1D("Final Triggers","Final Triggers", 1000,0,1000);
 randTrig = dbe_->book1D("Random Triggers","Random Triggers", 1000,0,1000);
 numberResets = dbe_->book1D("Number Resets","Number Resets", 1000,0,1000);
 deadTime = dbe_->book1D("DeadTime","DeadTime", 1000,0,1000);
 lostFinalTriggers = dbe_->book1D("Lost Final Trigger","Lost Final Trigger", 1000,0,1000);
 
 dbe_->setCurrentFolder("L1T/L1Scalers/L1TriggerRates");
 orbitNumRate = dbe_->book1D("Orbit Number Rate","Orbit Number Rate", 1000,0,1000);
 trigNumRate = dbe_->book1D("Trigger Number Rate","trigger Number Rate",1000,0,1000);
 eventNumRate = dbe_->book1D("event Number rate","event Number Rate", 1000,0,1000);
 finalTrigRate = dbe_->book1D("Final Trigger Rate","Final Trigger Rate", 1000,0,1000);
 randTrigRate = dbe_->book1D("Random Trigger Rate","Random Trigger Rate", 1000,0,1000);
 numberResetsRate = dbe_->book1D("Number Resets Rate","Number Resets Rate", 1000,0,1000);
 deadTimePercent = dbe_->book1D("DeadTimepercent","DeadTimePercent", 1000,0,1000);
 lostFinalTriggersPercent = dbe_->book1D("Lost Final Trigger Percent","Lost Final Triggerpercent", 1000,0,1000);

 dbe_->setCurrentFolder("L1T/L1Scalers/LumiScalers");
 instLumi = dbe_->book1D("Instant Lumi","Instant Lumi",1000,0,1000);
 instLumiErr = dbe_->book1D("Instant Lumi Err","Instant Lumi Err",1000,0,1000);
 instLumiQlty = dbe_->book1D("Instant Lumi Qlty","Instant Lumi Qlty",1000,0,1000);
 instEtLumi = dbe_->book1D("Instant Et Lumi","Instant Et Lumi",1000,0,1000);
 instEtLumiErr = dbe_->book1D("Instant Et Lumi Err","Instant Et Lumi Err",1000,0,1000);
 instEtLumiQlty = dbe_->book1D("Instant Et Lumi Qlty","Instant Et Lumi Qlty",1000,0,1000);
 sectionNum = dbe_->book1D("Section Number","Section Number",1000,0,1000);
 startOrbit = dbe_->book1D("Start Orbit","Start Orbit",1000,0,1000);
 numOrbits = dbe_->book1D("Num Orbits","Num Orbits",1000,0,1000);
 

 nev_=0;
} 




void L1Scalers::beginJob(const edm::EventSetup& iSetup)
{
  if ( verbose_ ) {
    std::cout << "L1Scalers::beginJob()..." << std::endl;
  }

  if (dbe_) {
    if ( verbose_ ) {
      dbe_->setVerbose(1);
    }
    dbe_->setCurrentFolder("L1T/L1Scalers");

  }
  
  
  return;
}

void L1Scalers::endJob(void)
{
if(outputFile_.size() != 0 && dbe_) dbe_->save(outputFile_);
}

void L1Scalers::analyze(const edm::Event &e, const edm::EventSetup &iSetup)
{
nev_++;
std::cout << "L1Scalers::analyze  event " << nev_ <<  std::endl;

edm::Handle<L1TriggerScalersCollection> triggerScalers;
  bool a = e.getByLabel(scalersSource_, triggerScalers);
  if ( !a ) {
    if ( verbose_ ) {
      std::cout << "L1Scalers::analyze: getByLabel failed with label " 
                << scalersSource_
                << std::endl;;
    }
    return;
  }

edm::Handle<L1TriggerRatesCollection> triggerRates;
  bool b = e.getByLabel(scalersSource_, triggerRates);
  if ( !b ) {
    if ( verbose_ ) {
      std::cout << "L1Scalers::analyze: getByLabel failed with label " 
                << scalersSource_
                << std::endl;;
    }
    return;
  }

edm::Handle<LumiScalersCollection> lumiScalers;
  bool c = e.getByLabel(scalersSource_, lumiScalers);
  if ( !c ) {
    if ( verbose_ ) {
      std::cout << "L1Scalers::analyze: getByLabel failed with label " 
                << scalersSource_
                << std::endl;;
    }
    return;
  }
  
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

}

void L1Scalers::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
				    const edm::EventSetup& iSetup)
{

}


/// BeginRun
void L1Scalers::beginRun(const edm::Run& run, const edm::EventSetup& iSetup)
{
  if ( verbose_) {
    std::cout << "L1Scalers::beginRun "<< std::endl;
  }
}

/// EndRun
void L1Scalers::endRun(const edm::Run& run, const edm::EventSetup& iSetup)
{
  if ( verbose_) {
    std::cout << "L1Scalers::endRun "<< std::endl;
  }
}


