// Class:      L1TScalersSCAL
// user include files

#include <sstream>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/Scalers/interface/L1TriggerScalers.h"
#include "DataFormats/Scalers/interface/L1TriggerRates.h"
#include "DataFormats/Scalers/interface/LumiScalers.h"
#include "DataFormats/Scalers/interface/L1AcceptBunchCrossing.h"
#include "DataFormats/Scalers/interface/ScalersRaw.h"

#include "DQM/TrigXMonitor/interface/L1TScalersSCAL.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DQMServices/Core/interface/DQMStore.h"


using namespace edm;

L1TScalersSCAL::L1TScalersSCAL(const edm::ParameterSet& ps):
  dbe_(0),
  scalersSource_( ps.getParameter< edm::InputTag >("scalersResults")),
  verbose_(ps.getUntrackedParameter <bool> ("verbose", false))
{
  LogDebug("Status") << "constructor" ;

  for ( int i=0; i<ScalersRaw::N_L1_TRIGGERS_v1; i++) 
    { bufferBits_.push_back(0); algorithmRates_.push_back(0);}
  bufferOrbitNumber_ = 0;
  
  dbe_ = Service<DQMStore>().operator->();
  if(dbe_) {
    dbe_->setVerbose(0);
    //dbe_->setCurrentFolder("L1T/L1TScalersSCAL");
        
    dbe_->setCurrentFolder("L1T/L1TScalersSCAL/L1TriggerScalers");
        
    orbitNum = dbe_->book1D("Orbit Number","Orbit Number", 1000,0,10E8);
    trigNum = dbe_->book1D("Trigger Number","Trigger Number",1000,0,100000);
    eventNum = dbe_->book1D("Event Number","Event Number", 1000,0,100000);
    finalTrig = dbe_->book1D("Final Triggers","Final Triggers", 1000,0,100000);
    randTrig = dbe_->book1D("Random Triggers","Random Triggers", 1000,0,1000);
    numberResets = dbe_->book1D("Number Resets","Number Resets", 1000,0,1000);
    deadTime = dbe_->book1D("DeadTime","DeadTime", 1000,0,100000);
    lostFinalTriggers = dbe_->book1D("Lost Final Trigger","Lost Final Trigger",
    				     1000,0,100000);

    char hname[40];//histo name
    char mename[40];//ME name

    dbe_->setCurrentFolder("L1T/L1TScalersSCAL/L1TriggerScalers/AlgorithmRates");

    for(int i=0; i<128; i++) {
      sprintf(hname, "Rate_AlgoBit_%03d", i);
      sprintf(mename, "Rate_AlgoBit _%03d", i);

      algoRate[i] = dbe_->book1D(hname, mename,101,-0.5,100.5);
      algoRate[i]->setAxisTitle("Lumi Section" ,1);
    }    				     
/*   
    dbe_->setCurrentFolder("L1T/L1TScalersSCAL/L1TriggerRates");
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
*/
    				
    dbe_->setCurrentFolder("L1T/L1TScalersSCAL/LumiScalers");
    instLumi = dbe_->book1D("Instant Lumi","Instant Lumi",100,0,100);
    instLumiErr = dbe_->book1D("Instant Lumi Err","Instant Lumi Err",100,
			       0,100);
    instLumiQlty = dbe_->book1D("Instant Lumi Qlty","Instant Lumi Qlty",100,
				0,100);
    instEtLumi = dbe_->book1D("Instant Et Lumi","Instant Et Lumi",100,0,100);
    instEtLumiErr = dbe_->book1D("Instant Et Lumi Err","Instant Et Lumi Err",
				 100,0,100);
    instEtLumiQlty = dbe_->book1D("Instant Et Lumi Qlty",
				  "Instant Et Lumi Qlty",100,0,100);
    sectionNum = dbe_->book1D("Section Number","Section Number",100,0,100);
    startOrbit = dbe_->book1D("Start Orbit","Start Orbit",100,0,100);
    numOrbits = dbe_->book1D("Num Orbits","Num Orbits",100,0,100);
        
    
    dbe_->setCurrentFolder("L1T/L1TScalersSCAL/L1AcceptBunchCrossing");
    orbitNumL1A = dbe_->book1D("Orbit Number L1A","OrbitNumber L1A",200,0,10E8);       
    bunchCrossingL1A = dbe_->book1D("Bunch Crossing L1A", "Bunch Crossing L1A", 3564, -0.5, 3563.5);    
    eventType = dbe_->book1D("Event Type", "Event Type", 8, -0.5, 7.5);
  }    


}


L1TScalersSCAL::~L1TScalersSCAL()
{
 
   // do anything here that needs to be done at destruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
L1TScalersSCAL::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   

   //access SCAL info
   edm::Handle<L1TriggerScalersCollection> triggerScalers;
   bool a = iEvent.getByLabel(scalersSource_, triggerScalers);
   //edm::Handle<L1TriggerRatesCollection> triggerRates;
   //bool b = iEvent.getByLabel(scalersSource_, triggerRates);
   edm::Handle<LumiScalersCollection> lumiScalers;
   bool c = iEvent.getByLabel(scalersSource_, lumiScalers);
   edm::Handle<L1AcceptBunchCrossingCollection> bunchCrossings;
   bool d = iEvent.getByLabel(scalersSource_, bunchCrossings);
   
   //if ( ! (a && b && c ) ) {
   if ( ! (a && c && d) ) {
      LogInfo("Status") << "getByLabel failed with label " 
       		        << scalersSource_;

   }
        		        
   else { // we have the data 

     L1TriggerScalersCollection::const_iterator it = triggerScalers->begin();
     if(triggerScalers->size()){ 

	 unsigned int lumisection = it->luminositySection();
//         std::cout << "lumisection " << lumisection << std::endl; 

	 if(lumisection){
           orbitNum ->Fill(it->orbitNumber());        
           trigNum ->Fill(it->triggerNumber());
	   eventNum ->Fill(it->eventNumber());
 	   finalTrig ->Fill(it->finalTriggersDistributed());
	   randTrig ->Fill(it->randomTriggers());
	   numberResets ->Fill(it->numberResets());
	   deadTime ->Fill(it->deadTime());
//           std::cout <<"dead time " << it->deadTime() << std::endl;

	   lostFinalTriggers ->Fill(it->lostFinalTriggers());                                    

           std::vector<unsigned int> algoBits = it->triggers();
//	   unsigned int orbitN = it->orbitNumber();
//	   std::vector<float> algoBitsF(algoBits.begin(), algoBits.end());

//	   if(bufferOrbitNumber_ != orbitN){
//	     std::cout << "OrbitNumber has changed!!!!" << std::endl;
//	     std::cout << "bufferOrbitNumber " << bufferOrbitNumber_ << std::endl;
//	     bufferOrbitNumber_ = orbitN;
//           std::cout << "orbitNum " << orbitN << std::endl; 
//	   }
         
	   if(bufferBits_ != algoBits){ 
  
	     bufferBits_ = algoBits;

//           for ( int i=0; i<ScalersRaw::N_L1_TRIGGERS_v1; i++){ 
             for (unsigned int i=0; i< algorithmRates_.size(); i++){ 


//               std::cout << "algobits"<< i << " "  << algoBits[i] << std::endl;
	       algorithmRates_[i] = (float) algoBits[i]/N_LUMISECTION_TIME ;
//             std::cout << "algorithmRates_"<< i << " "  << algorithmRates_[i] << std::endl;
               algoRate[i]->setBinContent(it->luminositySection()+1, algorithmRates_[i]); 
	     } 	   
           }
	 //std::cout << "algoBits.size = " << algoBits.size() << std::endl;
	 //std::cout << "bufferBits_.size = " << bufferBits_.size() << std::endl;

         /*
	 int length = algoBits.size() / 4;
         char line[128];
  	 for ( int i=0; i<length; i++)
  	 {
    	  sprintf(line," %3.3d: %10u    %3.3d: %10u    %3.3d: %10u    %3.3d: %10u",
	 	  i,              algoBits[i], 
	    	  (i+length),     algoBits[i+length], 
		  (i+(length*2)), algoBits[i+(length*2)], 
	    	  (i+(length*3)), algoBits[i+(length*3)]);
	    std::cout << line << std::endl;
  	 }

         sprintf(line,
	         " LuminositySection: %15d  BunchCrossingErrors:      %15d",
 	         it->luminositySection(), it->bunchCrossingErrors());
         std::cout << line << std::endl;
         */
         }
     }
/*
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
*/

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
       
       /*
       char line2[128];
       sprintf(line2," InstantLumi:       %e  Err: %e  Qlty: %d",
          it3->instantLumi(), it3->instantLumiErr(), it3->instantLumiQlty());
       std::cout << line2 << std::endl;

       sprintf(line2," SectionNumber: %10d   StartOrbit: %10d  NumOrbits: %10d",
   	  it3->sectionNumber(), it3->startOrbit(), it3->numOrbits());
       std::cout << line2 << std::endl;
       */
     }

     L1AcceptBunchCrossingCollection::const_iterator it4 = bunchCrossings->begin();

     if(bunchCrossings->size()){

       orbitNumL1A->Fill(it4->orbitNumber());
       bunchCrossingL1A->Fill(it4->bunchCrossing());
       eventType->Fill(it4->eventType());
	
     }
   
   } // getByLabel succeeds for scalers
       


}


// ------------ method called once each job just before starting event loop  ------------
void 
L1TScalersSCAL::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
L1TScalersSCAL::endJob() {
}


void L1TScalersSCAL::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
				   const edm::EventSetup& iSetup)
{				    
}
				    
/// BeginRun
void L1TScalersSCAL::beginRun(const edm::Run& run, const edm::EventSetup& iSetup)
{
}
				    
/// EndRun
void L1TScalersSCAL::endRun(const edm::Run& run, const edm::EventSetup& iSetup)
{
}
				    
