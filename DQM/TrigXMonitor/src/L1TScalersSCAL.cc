// Class:      L1TScalersSCAL
// user include files

#include <sstream>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Scalers/interface/Level1TriggerScalers.h"
#include "DataFormats/Scalers/interface/Level1TriggerRates.h"
#include "DataFormats/Scalers/interface/LumiScalers.h"
#include "DataFormats/Scalers/interface/L1AcceptBunchCrossing.h"
#include "DataFormats/Scalers/interface/ScalersRaw.h"
#include "DataFormats/Scalers/interface/TimeSpec.h"

#include "DQM/TrigXMonitor/interface/L1TScalersSCAL.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DQMServices/Core/interface/DQMStore.h"


const double SECS_PER_LUMI = 23.31040958083832;


using namespace edm;
using namespace std;


L1TScalersSCAL::L1TScalersSCAL(const edm::ParameterSet& ps):
  dbe_(0),
  scalersSource_( ps.getParameter< edm::InputTag >("scalersResults")),
  verbose_(ps.getUntrackedParameter <bool> ("verbose", false)),
  denomIsTech_(ps.getUntrackedParameter <bool> ("denomIsTech", true)),
  denomBit_(ps.getUntrackedParameter <unsigned int> ("denomBit", 40)),
  muonBit_(ps.getUntrackedParameter <unsigned int> ("muonBit", 55)),
  egammaBit_(ps.getUntrackedParameter <unsigned int> ("egammaBit", 46)),
  jetBit_(ps.getUntrackedParameter <unsigned int> ("jetBit", 15))
{
  LogDebug("Status") << "constructor" ;

  for ( int i=0; i<Level1TriggerScalers::nLevel1Triggers; i++) 
    { bufferAlgoRates_.push_back(0); algorithmRates_.push_back(0); integral_algo_.push_back(0.);}
  for ( int i=0; i<Level1TriggerScalers::nLevel1TestTriggers; i++) 
    { bufferTechRates_.push_back(0); technicalRates_.push_back(0); integral_tech_.push_back(0.);}

  buffertime_ = 0;
  reftime_ = 0;
  nev_ = 0;
  integral_tech_42_OR_43_ = 0;
  bufferLumi_ = 0;

  int maxNbins = 2001;

  dbe_ = Service<DQMStore>().operator->();
  if(dbe_) {
    dbe_->setVerbose(0);
        
    dbe_->setCurrentFolder("L1T/L1TScalersSCAL/Level1TriggerScalers");
        
    //orbitNum = dbe_->book1D("Orbit_Number","Orbit_Number", 1000,0,10E8);
    orbitNum = dbe_->book1D("Orbit_Number","Orbit_Number", maxNbins,-0.5,double(maxNbins)-0.5);

    trigNum = dbe_->book1D("Number_of_Triggers","Number_of_Triggers",1000,0,4E4);
    trigNum->setAxisTitle("Time [sec]", 1);

    //eventNum = dbe_->book1D("Number_of_Events","Number_of_Events", 1000,0,1E7);
    eventNum = dbe_->book1D("Number_of_Events","Number_of_Events", maxNbins,-0.5,double(maxNbins)-0.5);

    physTrig = dbe_->book1D("Physics_Triggers","Physics_Triggers", maxNbins,-0.5,double(maxNbins)-0.5);
    physTrig->setAxisTitle("Lumi Section", 1);

    randTrig = dbe_->book1D("Random_Triggers","Random_Triggers", maxNbins,-0.5,double(maxNbins)-0.5);
    randTrig->setAxisTitle("Lumi Section", 1);

    //numberResets = dbe_->book1D("Number_Resets","Number_Resets", 1000,0,1000);
    numberResets = dbe_->book1D("Number_Resets","Number_Resets", maxNbins,-0.5,double(maxNbins)-0.5);
   
    //deadTime = dbe_->book1D("DeadTime","DeadTime", 1000,0,1E9);
    deadTime = dbe_->book1D("DeadTime","DeadTime", maxNbins,-0.5,double(maxNbins)-0.5);
    
    //lostFinalTriggers = dbe_->book1D("Lost_Final_Trigger","Lost_Final_Trigger", 1000,0,1E6);
    lostFinalTriggers = dbe_->book1D("Lost_Final_Trigger","Lost_Final_Trigger", maxNbins,-0.5,double(maxNbins)-0.5);
   
    dbe_->setCurrentFolder("L1T/L1TScalersSCAL/Level1TriggerRates");

    physRate = dbe_->book1D("Physics_Trigger_Rate","Physics_Trigger_Rate", 
			    maxNbins,-0.5,double(maxNbins)-0.5);
    //physRate->setAxisTitle("Lumi Section", 1);

    randRate = dbe_->book1D("Random_Trigger_Rate","Random_Trigger_Rate", 
			    maxNbins,-0.5,double(maxNbins)-0.5);
    //randRate->setAxisTitle("Lumi Section", 1);

    deadTimePercent = dbe_->book1D("Deadtime_Percent","Deadtime_Percent", 
    				   maxNbins,-0.5,double(maxNbins)-0.5);
    //deadTimePercent->setAxisTitle("Lumi Section", 1);

    lostPhysRate = dbe_->book1D("Lost_Physics_Trigger_Rate","Lost_Physics_Trigger_Rate", 
				maxNbins,-0.5,double(maxNbins)-0.5);
    //lostPhysRate->setAxisTitle("Lumi Section", 1);

    lostPhysRateBeamActive = dbe_->book1D("Lost_Physics_Trigger_Rate_Beam_Active",
					  "Lost_Physics_Trigger_Rate_Beam_Active", 
					  maxNbins,-0.5,double(maxNbins)-0.5);
    //lostPhysRateBeamActive->setAxisTitle("Lumi Section", 1);

    
    instTrigRate = dbe_->book1D("instTrigRate","instTrigRate",1000,0,4E4);
    instTrigRate->setAxisTitle("Time [sec]", 1);

    instEventRate = dbe_->book1D("instEventRate","instEventRate",1000,0,4E4);
    instEventRate->setAxisTitle("Time [sec]", 1);

    char hname[40];//histo name
    char mename[40];//ME name

    dbe_->setCurrentFolder("L1T/L1TScalersSCAL/Level1TriggerRates/AlgorithmRates");

    for(int i=0; i<Level1TriggerScalers::nLevel1Triggers; i++) {
      sprintf(hname, "Rate_AlgoBit_%03d", i);
      sprintf(mename, "Rate_AlgoBit _%03d", i);

      algoRate[i] = dbe_->book1D(hname, mename,maxNbins,-0.5,double(maxNbins)-0.5);
      algoRate[i]->setAxisTitle("Lumi Section" ,1);
    }    
    				     
    dbe_->setCurrentFolder("L1T/L1TScalersSCAL/Level1TriggerRates/AlgorithmRates/Integrated");

    for(int i=0; i<Level1TriggerScalers::nLevel1Triggers; i++) {
      sprintf(hname, "Integral_AlgoBit_%03d", i);
      sprintf(mename, "Integral_AlgoBit _%03d", i);

      integralAlgo[i] = dbe_->book1D(hname, mename,maxNbins,-0.5,double(maxNbins)-0.5);
      integralAlgo[i]->setAxisTitle("Lumi Section" ,1);
    }    				     

    dbe_->setCurrentFolder("L1T/L1TScalersSCAL/Level1TriggerRates/TechnicalRates");

    for(int i=0; i<Level1TriggerScalers::nLevel1TestTriggers; i++) {
      sprintf(hname, "Rate_TechBit_%03d", i);
      sprintf(mename, "Rate_TechBit _%03d", i);

      techRate[i] = dbe_->book1D(hname, mename,maxNbins,-0.5,double(maxNbins)-0.5);
      techRate[i]->setAxisTitle("Lumi Section" ,1);
    }

    dbe_->setCurrentFolder("L1T/L1TScalersSCAL/Level1TriggerRates/TechnicalRates/Integrated");

    for(int i=0; i<Level1TriggerScalers::nLevel1TestTriggers; i++) {
      sprintf(hname, "Integral_TechBit_%03d", i);
      sprintf(mename, "Integral_TechBit _%03d", i);

      integralTech[i] = dbe_->book1D(hname, mename,maxNbins,-0.5,double(maxNbins)-0.5);
      integralTech[i]->setAxisTitle("Lumi Section" ,1);
    }    				     
    integralTech_42_OR_43 = dbe_->book1D("Integral_TechBit_042_OR_043","Integral_TechBit _042_OR_043",maxNbins,-0.5,double(maxNbins)-0.5);
    integralTech_42_OR_43->setAxisTitle("Lumi Section" ,1);


    dbe_->setCurrentFolder("L1T/L1TScalersSCAL/Level1TriggerRates/Ratios");

    std::stringstream smu,seg,sjet,sdenom;
    //denominator string
    if(denomIsTech_) sdenom << "_TechBit_";
    else sdenom << "_AlgoBit_";
    sdenom << denomBit_;
    //Muon ratio
    smu << muonBit_;
    rateRatio_mu = dbe_->book1D("Rate_Ratio_mu_PhysBit_"+ smu.str()+sdenom.str(), "Rate_Ratio_mu_PhysBit_" + smu.str()+sdenom.str(),maxNbins,-0.5,double(maxNbins)-0.5);
    //rateRatio_mu->setAxisTitle("Lumi Section" , 1);
    //Egamma ratio
    seg << egammaBit_;
    rateRatio_egamma = dbe_->book1D("Rate_Ratio_egamma_PhysBit_"+ seg.str()+sdenom.str(), "Rate_Ratio_egamma_PhysBit_" + seg.str()+sdenom.str(),maxNbins,-0.5,double(maxNbins)-0.5);
    //rateRatio_egamma->setAxisTitle("Lumi Section" , 1);
    //Jet ratio
    sjet << jetBit_;
    rateRatio_jet = dbe_->book1D("Rate_Ratio_jet_PhysBit_" + sjet.str()+sdenom.str(), "Rate_Ratio_jet_PhysBit_" + sjet.str()+sdenom.str(),maxNbins,-0.5,double(maxNbins)-0.5);
    //rateRatio_jet->setAxisTitle("Lumi Section" , 1);

    //HF bit ratios
    techRateRatio_8 = dbe_->book1D("Rate_Ratio_TechBit_8"+sdenom.str(), "Rate_Ratio_TechBit_8"+sdenom.str(),maxNbins,-0.5,double(maxNbins)-0.5);
    //techRateRatio_8->setAxisTitle("Lumi Section" ,1);
    techRateRatio_9 = dbe_->book1D("Rate_Ratio_TechBit_9"+sdenom.str(), "Rate_Ratio_TechBit_9"+sdenom.str(),maxNbins,-0.5,double(maxNbins)-0.5);
    //techRateRatio_9->setAxisTitle("Lumi Section" ,1);
    techRateRatio_10 = dbe_->book1D("Rate_Ratio_TechBit_10"+sdenom.str(), "Rate_Ratio_TechBit_10"+sdenom.str(),maxNbins,-0.5,double(maxNbins)-0.5);
    //techRateRatio_10->setAxisTitle("Lumi Section" ,1);

    //Other tech bit ratios
    techRateRatio_33_over_32 = dbe_->book1D("Rate_Ratio_TechBits_33_over_32", "Rate_Ratio_TechBits_33_over_32",maxNbins,-0.5,double(maxNbins)-0.5);
    techRateRatio_36 = dbe_->book1D("Rate_Ratio_TechBit_36"+sdenom.str(), "Rate_Ratio_TechBit_36"+sdenom.str(),maxNbins,-0.5,double(maxNbins)-0.5);
    techRateRatio_37 = dbe_->book1D("Rate_Ratio_TechBit_37"+sdenom.str(), "Rate_Ratio_TechBit_37"+sdenom.str(),maxNbins,-0.5,double(maxNbins)-0.5);
    techRateRatio_38 = dbe_->book1D("Rate_Ratio_TechBit_38"+sdenom.str(), "Rate_Ratio_TechBit_38"+sdenom.str(),maxNbins,-0.5,double(maxNbins)-0.5);
    techRateRatio_39 = dbe_->book1D("Rate_Ratio_TechBit_39"+sdenom.str(), "Rate_Ratio_TechBit_39"+sdenom.str(),maxNbins,-0.5,double(maxNbins)-0.5);
    techRateRatio_40 = dbe_->book1D("Rate_Ratio_TechBit_40"+sdenom.str(), "Rate_Ratio_TechBit_40"+sdenom.str(),maxNbins,-0.5,double(maxNbins)-0.5);
    techRateRatio_41 = dbe_->book1D("Rate_Ratio_TechBit_41"+sdenom.str(), "Rate_Ratio_TechBit_41"+sdenom.str(),maxNbins,-0.5,double(maxNbins)-0.5);
    techRateRatio_42 = dbe_->book1D("Rate_Ratio_TechBit_42"+sdenom.str(), "Rate_Ratio_TechBit_42"+sdenom.str(),maxNbins,-0.5,double(maxNbins)-0.5);
    techRateRatio_43 = dbe_->book1D("Rate_Ratio_TechBit_43"+sdenom.str(), "Rate_Ratio_TechBit_43"+sdenom.str(),maxNbins,-0.5,double(maxNbins)-0.5);

    				
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

      sprintf(hname, "OrbitNumber_L1A_%d", i+1);
      sprintf(mename, "OrbitNumber_L1A_%d", i+1);
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
  
  double evtLumi = iEvent.luminosityBlock();
  int run = iEvent.id().run();


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

	//orbitNum->Fill(it->orbitNr());
	orbitNum->setBinContent(lumisection+1,it->orbitNr());

	//trigNum ->Fill(it->gtTriggers());
	//trigNum->setBinContent(lumisection+1, it->gtTriggers()); 

	//eventNum->Fill(it->gtEvents());
	eventNum->setBinContent(lumisection+1,it->gtEvents());

	//physTrig ->Fill(it->l1AsPhysics());
	physTrig->setBinContent(lumisection+1, it->l1AsPhysics()); 

	//randTrig ->Fill(it->l1AsRandom());
	randTrig->setBinContent(lumisection+1, it->l1AsRandom()); 

	//numberResets ->Fill(it->gtResets());	 
	numberResets->setBinContent(lumisection+1, it->gtResets());	 

	//deadTime ->Fill(it->deadtime());
	deadTime->setBinContent(lumisection+1, it->deadtime());

	//lostFinalTriggers ->Fill(it->triggersPhysicsLost());                       
	lostFinalTriggers->setBinContent(lumisection+1, it->triggersPhysicsLost());
	 
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

	Level1TriggerRates trigRates(*it,run);
	Level1TriggerRates *triggerRates = &trigRates;
	if(triggerRates){
	  algorithmRates_ = triggerRates->gtAlgoCountsRate();
	  technicalRates_ = triggerRates->gtTechCountsRate();
	  
	  if( ((bufferLumi_!=lumisection) && (bufferLumi_<lumisection) && (evtLumi>1 || evtLumi==lumisection+1)) ){
	    bufferLumi_ = lumisection;
 
	    if(bufferAlgoRates_ != algorithmRates_){ 
	      bufferAlgoRates_ = algorithmRates_;
	      for (unsigned int i=0; i< algorithmRates_.size(); i++){ 
		integral_algo_[i]+=(algorithmRates_[i]*SECS_PER_LUMI);
		algoRate[i]->setBinContent(lumisection+1, algorithmRates_[i]); 
		integralAlgo[i]->setBinContent(lumisection+1,integral_algo_[i]); 
	      } 	   
	    }
	    if(bufferTechRates_ != technicalRates_){ 
	      bufferTechRates_ = technicalRates_;
	      for (unsigned int i=0; i< technicalRates_.size(); i++){ 
		integral_tech_[i]+=(technicalRates_[i]*SECS_PER_LUMI);
		techRate[i]->setBinContent(lumisection+1, technicalRates_[i]); 
		integralTech[i]->setBinContent(lumisection+1, integral_tech_[i]); 
		if( (i==42 || i==43) ) integral_tech_42_OR_43_+=(technicalRates_[i]*SECS_PER_LUMI);
	      } 

	      //fill rate ratio plots
	      if(denomIsTech_){
		if( denomBit_ < technicalRates_.size() ){
		  if(technicalRates_[denomBit_]){

		    if( muonBit_ < algorithmRates_.size() )
		      rateRatio_mu->setBinContent(lumisection+1, algorithmRates_[muonBit_]/technicalRates_[denomBit_]);
		    if( egammaBit_ < algorithmRates_.size() )
		      rateRatio_egamma->setBinContent(lumisection+1, algorithmRates_[egammaBit_]/technicalRates_[denomBit_]);
		    if( jetBit_ < algorithmRates_.size() )
		      rateRatio_jet->setBinContent(lumisection+1, algorithmRates_[jetBit_]/technicalRates_[denomBit_]);

		    techRateRatio_8->setBinContent(lumisection+1, technicalRates_[8]/technicalRates_[denomBit_]);
		    techRateRatio_9->setBinContent(lumisection+1, technicalRates_[9]/technicalRates_[denomBit_]);
		    techRateRatio_10->setBinContent(lumisection+1, technicalRates_[10]/technicalRates_[denomBit_]);

		    techRateRatio_36->setBinContent(lumisection+1, technicalRates_[36]/technicalRates_[denomBit_]);
		    techRateRatio_37->setBinContent(lumisection+1, technicalRates_[37]/technicalRates_[denomBit_]);
		    techRateRatio_38->setBinContent(lumisection+1, technicalRates_[38]/technicalRates_[denomBit_]);
		    techRateRatio_39->setBinContent(lumisection+1, technicalRates_[39]/technicalRates_[denomBit_]);
		    techRateRatio_40->setBinContent(lumisection+1, technicalRates_[40]/technicalRates_[denomBit_]);
		    techRateRatio_41->setBinContent(lumisection+1, technicalRates_[41]/technicalRates_[denomBit_]);
		    techRateRatio_42->setBinContent(lumisection+1, technicalRates_[42]/technicalRates_[denomBit_]);
		    techRateRatio_43->setBinContent(lumisection+1, technicalRates_[43]/technicalRates_[denomBit_]);
		  }
		}
	      }
	      if( technicalRates_[32]!=0 ) techRateRatio_33_over_32->setBinContent(lumisection+1, technicalRates_[33]/technicalRates_[32]);
	      integralTech_42_OR_43->setBinContent(lumisection+1, integral_tech_42_OR_43_);	   

	    }

	    physRate->setBinContent(lumisection+1, 
				    triggerRates->l1AsPhysicsRate()); 
	    randRate->setBinContent(lumisection+1, 
				    triggerRates->l1AsRandomRate());
	    lostPhysRate->setBinContent(lumisection+1, 
					triggerRates->triggersPhysicsLostRate()); 
	    lostPhysRateBeamActive->setBinContent(lumisection+1, 
						  triggerRates->triggersPhysicsLostBeamActiveRate()); 
	    deadTimePercent->setBinContent(lumisection+1, 
					   triggerRates->deadtimePercent()); 
	   
	  }//bufferLumi test
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
L1TScalersSCAL::beginJob(void)
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
				    
