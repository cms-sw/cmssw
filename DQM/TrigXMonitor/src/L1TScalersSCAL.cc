// Class:      L1TScalersSCAL
// user include files

#include <sstream>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/Scalers/interface/Level1TriggerScalers.h"
#include "DataFormats/Scalers/interface/Level1TriggerRates.h"
#include "DataFormats/Scalers/interface/LumiScalers.h"
#include "DataFormats/Scalers/interface/L1AcceptBunchCrossing.h"
#include "DataFormats/Scalers/interface/ScalersRaw.h"
#include "DataFormats/Scalers/interface/TimeSpec.h"

#include "DQM/TrigXMonitor/interface/L1TScalersSCAL.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DQMServices/Core/interface/DQMStore.h"


using namespace edm;
using namespace std;


L1TScalersSCAL::L1TScalersSCAL(const edm::ParameterSet& ps):
  dbe_(0),
  scalersSource_( ps.getParameter< edm::InputTag >("scalersResults")),
  verbose_(ps.getUntrackedParameter <bool> ("verbose", false))
{
  LogDebug("Status") << "constructor" ;

  for ( int i=0; i<Level1TriggerScalers::nLevel1Triggers; i++) 
    { bufferAlgoRates_.push_back(0); algorithmRates_.push_back(0);}
  for ( int i=0; i<Level1TriggerScalers::nLevel1TestTriggers; i++) 
    { bufferTechRates_.push_back(0); technicalRates_.push_back(0);}

  buffertime_ = 0;
  reftime_ = 0;
  nev_ = 0;

  dbe_ = Service<DQMStore>().operator->();
  if(dbe_) {
    dbe_->setVerbose(0);
        
    dbe_->setCurrentFolder("L1T/L1TScalersSCAL/Level1TriggerScalers");
        
    orbitNum = dbe_->book1D("Orbit Number","Orbit Number", 1000,0,10E8);
    trigNum = dbe_->book1D("Number of Triggers","Number of Triggers",1000,0,3E4);
    trigNum->setAxisTitle("Time [sec]", 1);

    eventNum = dbe_->book1D("Number of Events","Number of Events", 1000,0,1E7);

    physTrig = dbe_->book1D("Physics Triggers","Physics Triggers", 101,-0.5,100.5);
    physTrig->setAxisTitle("Lumisection", 1);

    randTrig = dbe_->book1D("Random Triggers","Random Triggers", 101,-0.5,100.5);
    randTrig->setAxisTitle("Lumisection", 1);

    numberResets = dbe_->book1D("Number Resets","Number Resets", 1000,0,1000);
    deadTime = dbe_->book1D("DeadTime","DeadTime", 1000,0,1E9);
    lostFinalTriggers = dbe_->book1D("Lost Final Trigger","Lost Final Trigger",
    				     1000,0,1E6);
   
    dbe_->setCurrentFolder("L1T/L1TScalersSCAL/Level1TriggerRates");

    physRate = dbe_->book1D("Physics Trigger Rate","Physics Trigger Rate", 
			    101,-0.5,100.5);
    physRate->setAxisTitle("Lumisection", 1);

    randRate = dbe_->book1D("Random Trigger Rate","Random Trigger Rate", 
			    101,-0.5,100.5);
    randRate->setAxisTitle("Lumisection", 1);

    deadTimePercent = dbe_->book1D("Deadtime Percent","Deadtime Percent", 
    				   101,-0.5,100.5);
    deadTimePercent->setAxisTitle("Lumisection", 1);

    lostPhysRate = dbe_->book1D("Lost Physics Trigger Rate","Lost Physics Trigger Rate", 
				101,-0.5,100.5);
    lostPhysRate->setAxisTitle("Lumisection", 1);

    lostPhysRateBeamActive = dbe_->book1D("Lost Physics Trigger Rate - Beam Active",
					  "Lost Physics Trigger Rate - Beam Active", 
					  101,-0.5,100.5);
    lostPhysRateBeamActive->setAxisTitle("Lumisection", 1);

    
    instTrigRate = dbe_->book1D("instTrigRate","Instantaneous Trigger Rate",1000,0,3E4);
    instTrigRate->setAxisTitle("Time [sec]", 1);

    instEventRate = dbe_->book1D("instEventRate","Instantaneous Event Rate",1000,0,3E4);
    instEventRate->setAxisTitle("Time [sec]", 1);

    char hname[40];//histo name
    char mename[40];//ME name

    dbe_->setCurrentFolder("L1T/L1TScalersSCAL/Level1TriggerRates/AlgorithmRates");

    for(int i=0; i<Level1TriggerScalers::nLevel1Triggers; i++) {
      sprintf(hname, "Rate_AlgoBit_%03d", i);
      sprintf(mename, "Rate_AlgoBit _%03d", i);

      algoRate[i] = dbe_->book1D(hname, mename,101,-0.5,100.5);
      algoRate[i]->setAxisTitle("Lumi Section" ,1);
    }    				     
    dbe_->setCurrentFolder("L1T/L1TScalersSCAL/Level1TriggerRates/TechnicalRates");

    for(int i=0; i<Level1TriggerScalers::nLevel1TestTriggers; i++) {
      sprintf(hname, "Rate_TechBit_%03d", i);
      sprintf(mename, "Rate_TechBit _%03d", i);

      techRate[i] = dbe_->book1D(hname, mename,101,-0.5,100.5);
      techRate[i]->setAxisTitle("Lumi Section" ,1);
    }    				     

    				
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
       
    for(int i=0; i<4; i++){

      sprintf(hname, "Orbit_Number_L1A_%d", i+1);
      sprintf(mename, "Orbit_Number_L1A_%d", i+1);
      orbitNumL1A[i] = dbe_->book1D(hname,mename,200,0,10E8);          

      sprintf(hname, "Bunch_Crossing_L1A_%d", i+1);
      sprintf(mename, "Bunch_Crossing_L1A_%d", i+1);
      bunchCrossingL1A[i]= dbe_->book1D(hname, mename, 3564, -0.5, 3563.5);
    }
    orbitNumL1A[0]->setAxisTitle("Current BX",1);
    orbitNumL1A[1]->setAxisTitle("Previous BX",1);
    orbitNumL1A[2]->setAxisTitle("Second Previous BX",1);
    orbitNumL1A[3]->setAxisTitle("Third Previous BX",1);

    bunchCrossingL1A[0]->setAxisTitle("Current BX",1);
    bunchCrossingL1A[1]->setAxisTitle("Previous BX",1);
    bunchCrossingL1A[2]->setAxisTitle("Second Previous BX",1);
    bunchCrossingL1A[3]->setAxisTitle("Third Previous BX",1);


    for(int j=0; j<3; j++) {
      sprintf(hname, "BX_Correlation_%d", j+1);
      sprintf(mename, "BX_Correlation_%d", j+1);

      bunchCrossingCorr[j] = dbe_->book2D(hname, mename, 99,-0.5,3563.5, 99,-0.5,3563.5);
      bunchCrossingCorr[j]->setAxisTitle("Current Event", 1);

      sprintf(hname, "Bunch_Crossing_Diff_%d", j+1);
      sprintf(mename, "Bunch_Crossing_Diff_%d", j+1);
      
      bunchCrossingDiff[j] = dbe_->book1D(hname, mename, 1000,0,1E6);

      sprintf(hname, "Bunch_Crossing_Diff_small_%d", j+1);
      sprintf(mename, "Bunch_Crossing_Diff_small_%d", j+1);
      
      bunchCrossingDiff_small[j] = dbe_->book1D(hname, mename, 1000,0,1000);

    }    				     
    bunchCrossingCorr[0]->setAxisTitle("Previous Event" , 2);
    bunchCrossingCorr[1]->setAxisTitle("Second Previous Event" , 2);
    bunchCrossingCorr[2]->setAxisTitle("Third Previous Event" , 2);

    bunchCrossingDiff[0]->setAxisTitle("BX_Current - BX_Previous" , 1);
    bunchCrossingDiff[1]->setAxisTitle("BX_Current - BX_SecondPrevious" , 1);
    bunchCrossingDiff[2]->setAxisTitle("BX_Current - BX_ThirdPrevious" , 1);

    bunchCrossingDiff_small[0]->setAxisTitle("BX_Current - BX_Previous" , 1);
    bunchCrossingDiff_small[1]->setAxisTitle("BX_Current - BX_SecondPrevious" , 1);
    bunchCrossingDiff_small[2]->setAxisTitle("BX_Current - BX_ThirdPrevious" , 1);

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
  nev_++;
  
  //access SCAL info
  edm::Handle<Level1TriggerScalersCollection> triggerScalers;
  bool a = iEvent.getByLabel(scalersSource_, triggerScalers);
  //edm::Handle<L1TriggerRatesCollection> triggerRates;
  //edm::Handle<Level1TriggerRatesCollection> triggerRates;
  //bool b = iEvent.getByLabel(scalersSource_, triggerRates);
  edm::Handle<LumiScalersCollection> lumiScalers;
  bool c = iEvent.getByLabel(scalersSource_, lumiScalers);
  edm::Handle<L1AcceptBunchCrossingCollection> bunchCrossings;
  bool d = iEvent.getByLabel(scalersSource_, bunchCrossings);
  
  if ( ! (a && c && d) ) {
    LogInfo("Status") << "getByLabel failed with label " 
		      << scalersSource_;
  }
        		        
  else { // we have the data 
    
    Level1TriggerScalersCollection::const_iterator it = triggerScalers->begin();
    if(triggerScalers->size()){ 

      //for(L1TriggerScalersCollection::const_iterator it = triggerScalers->begin();
      //it != triggerScalers->end();
      //++it){
       
      unsigned int lumisection = it->lumiSegmentNr();
      struct timespec thetime = it->collectionTime();
      long currenttime;
      //cout << "lumisection = " << lumisection << endl;
      if(nev_ == 1) reftime_ = thetime.tv_sec; 
      //cout << "reftime = " << reftime_ << endl;
      if(lumisection){
	orbitNum ->Fill(it->orbitNr());
	//trigNum ->Fill(it->gtTriggers());
	//trigNum->setBinContent(lumisection+1, it->gtTriggers()); 
	eventNum ->Fill(it->gtEvents());
	//physTrig ->Fill(it->l1AsPhysics());
	physTrig->setBinContent(lumisection+1, it->l1AsPhysics()); 
	//randTrig ->Fill(it->l1AsRandom());
	randTrig->setBinContent(lumisection+1, it->l1AsRandom()); 
	numberResets ->Fill(it->gtResets());	 
	deadTime ->Fill(it->deadtime());
	lostFinalTriggers ->Fill(it->triggersPhysicsLost());                                    
	 
	//cout << "lumisection = " << lumisection << " , orbitNum = " 
	//     << it->orbitNr() << ", lumiSegmentOrbits = " << it->lumiSegmentOrbits() 
	//     << ", l1AsPhys = " << it->l1AsPhysics() << endl;
	//cout << "gtTriggersRate = " << it->gtTriggersRate() << endl;

	if(buffertime_ < thetime.tv_sec){
	  buffertime_ = thetime.tv_sec;
	  currenttime = thetime.tv_sec - reftime_ ;
	  int timebin = (int)(currenttime/30) + 1;
	  //cout << "time bin = " << timebin << endl;
	  trigNum->setBinContent((int)timebin ,it->gtTriggers());
	  instTrigRate->setBinContent((int)timebin ,it->gtTriggersRate());
	  instEventRate->setBinContent((int)timebin ,it->gtEventsRate());
	}
	
	//cout << "tv_sec = " << thetime.tv_sec << endl;

	//std::vector<unsigned int> algoBits = it->gtAlgoCounts();
	//std::vector<unsigned int> techBits = it->gtTechCounts();
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
	*/
	//sprintf(line,
	//        " LuminositySection: %15d  BunchCrossingErrors:      %15d",
	//        it->luminositySection(), it->bunchCrossingErrors());
	//std::cout << line << std::endl;
         
	//Level1TriggerRatesCollection::const_iterator it2 = triggerRates->begin();

	Level1TriggerRates triggerRates(*it);
	if(&triggerRates){
	  algorithmRates_ = triggerRates.gtAlgoCountsRate();
	  technicalRates_ = triggerRates.gtTechCountsRate();
	   
	  if(bufferAlgoRates_ != algorithmRates_){ 
	    bufferAlgoRates_ = algorithmRates_;
	    for (unsigned int i=0; i< algorithmRates_.size(); i++){ 
	      algoRate[i]->setBinContent(lumisection+1, algorithmRates_[i]); 
	    } 	   
	  }
	  if(bufferTechRates_ != technicalRates_){ 
	    bufferTechRates_ = technicalRates_;
	    for (unsigned int i=0; i< technicalRates_.size(); i++){ 
	      techRate[i]->setBinContent(lumisection+1, technicalRates_[i]); 
	    } 	   
	  }

	  physRate->setBinContent(lumisection+1, 
				  triggerRates.l1AsPhysicsRate()); 
	  randRate->setBinContent(lumisection+1, 
				  triggerRates.triggersPhysicsLostRate());
	  lostPhysRate->setBinContent(lumisection+1, 
				  triggerRates.l1AsPhysicsRate()); 
	  lostPhysRateBeamActive->setBinContent(lumisection+1, 
				  triggerRates.triggersPhysicsLostBeamActiveRate()); 
	  deadTimePercent->setBinContent(lumisection+1, 
				  triggerRates.deadtimePercent()); 
	   
	}//triggerRates	 
      }//lumisection
    }//triggerScalers->size()

     
    LumiScalersCollection::const_iterator it3 = lumiScalers->begin();
    if(lumiScalers->size()){ 
    
      //for(LumiScalersCollection::const_iterator it3 = lumiScalers->begin();
      //it3 != lumiScalers->end();
      //++it3){
       
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

    //     L1AcceptBunchCrossingCollection::const_iterator it4 = bunchCrossings->begin();
    //if(bunchCrossings->size()){
    //int counttest=0;
    int l1accept; 
    unsigned int bx_current = 0, orbitnumber_current = 0, bxdiff = 0;

    for(L1AcceptBunchCrossingCollection::const_iterator it4 = bunchCrossings->begin();
	it4 != bunchCrossings->end();
	++it4){
      //counttest++;
      //cout << "counttest = " << counttest << endl;
      //cout << "bunchCrossing = "  << it4->bunchCrossing() << endl;
      //cout << "l1AcceptOffset = " << it4->l1AcceptOffset() << endl;

      l1accept = abs(it4->l1AcceptOffset());   

      //cout << "l1a orbit number (before if)= " << it4->orbitNumber() << endl;
      //cout << "l1a bunch crossing = " << it4->bunchCrossing() << endl;

      if(l1accept == 0){
	orbitnumber_current = it4->orbitNumber();
	orbitNumL1A[l1accept]->Fill(orbitnumber_current);

	bx_current = it4->bunchCrossing();
	bunchCrossingL1A[l1accept]->Fill(bx_current);

      }
      else if (l1accept==1 || l1accept==2 || l1accept==3){
	orbitNumL1A[l1accept]->Fill(it4->orbitNumber());
	bunchCrossingL1A[l1accept]->Fill(it4->bunchCrossing());
	//cout << "l1accept = " << l1accept << ", bx_current = " << bx_current << ", it4->bunchCrossing() = " << it4->bunchCrossing() << endl;
	bunchCrossingCorr[l1accept-1]->Fill(bx_current, it4->bunchCrossing());
	bxdiff = 3564*(orbitnumber_current-it4->orbitNumber()) + bx_current - it4->bunchCrossing();
	bunchCrossingDiff[l1accept-1]->Fill(bxdiff);
	bunchCrossingDiff_small[l1accept-1]->Fill(bxdiff);
      }

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
				    
