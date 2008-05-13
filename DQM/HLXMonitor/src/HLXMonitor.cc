/*
    Author:  Adam Hunt
    email:   ahunt@princeton.edu
*/
#include "DQM/HLXMonitor/interface/HLXMonitor.h"

// STL Headers

#include <iomanip>
#include <TSystem.h>

using std::cout;
using std::endl;

HLXMonitor::HLXMonitor(const edm::ParameterSet& iConfig)
{

   NUM_HLX          = iConfig.getUntrackedParameter< unsigned int >("numHlx",     36);
   NUM_BUNCHES      = iConfig.getUntrackedParameter< unsigned int >("numBunches", 3564);
   listenPort       = iConfig.getUntrackedParameter< unsigned int >("SourcePort", 51001);
   OutputFilePrefix = iConfig.getUntrackedParameter< std::string  >("outputFile", "lumi");
   OutputDir        = iConfig.getUntrackedParameter< std::string  >("outputDir","  data");
   SavePeriod       = iConfig.getUntrackedParameter< unsigned int >("SavePeriod",  10);
   NBINS            = iConfig.getUntrackedParameter< unsigned int >("NBINS",       297);  // 12 BX per bin
   XMIN             = iConfig.getUntrackedParameter< double       >("XMIN",        0);
   XMAX             = iConfig.getUntrackedParameter< double       >("XMAX",        3564);
   Style            = iConfig.getUntrackedParameter< std::string  >("Style",       "BX");
   AquireMode       = iConfig.getUntrackedParameter< unsigned int >("AquireMode",  0);
   Accumulate       = iConfig.getUntrackedParameter< bool         >("Accumulate",  true); // all
   TriggerBX        = iConfig.getUntrackedParameter< unsigned int >("TriggerBX",   50);
   reconnTime       = iConfig.getUntrackedParameter< unsigned int >("ReconnectionTime",5);
   DistribIP        = iConfig.getUntrackedParameter< std::string  >("HLXDAQIP",    "vmepcs2f17-19");
   ResetAtNewRun    = iConfig.getUntrackedParameter< bool         >("NewRun_Reset","true");

   // HLX Config info
   set1BelowIndex   = 0;
   set1BetweenIndex = 1;
   set1AboveIndex   = 2;
   set2BelowIndex   = 3;
   set2BetweenIndex = 4;
   set2AboveIndex   = 5;

   runNumLength     = 9;
   secNumLength     = 6;

   runNumber_       = 0;

   if(NUM_HLX > 36)       
     NUM_HLX = 36;

   if(NUM_BUNCHES > 3564) 
     NUM_BUNCHES = 3564;

   if(XMAX <= XMIN){
     XMIN = 0;
     if(XMAX <= 0) XMAX = 3564;
   }
   
   if((Style.compare("History")==0) || (NBINS == 0)){
     NBINS = (unsigned int)(XMAX-XMIN);
   }

   dbe_ = edm::Service<DQMStore>().operator->();

   if ( dbe_ ) {
     dbe_->setVerbose(1);
   }

   monitorName_ = iConfig.getUntrackedParameter<std::string>("monitorName","HLX");
   cout << "Monitor name = " << monitorName_ << endl;
   prescaleEvt_ = iConfig.getUntrackedParameter<int>("prescaleEvt", -1);
   cout << "===>DQM event prescale = " << prescaleEvt_ << " events "<< endl;

   int HLXHFMapTemp[] = {31,32,33,34,35,18,  // s2f02 hf-
			 13,14,15,16,17,0,   // s2f02 hf+
			 25,26,27,28,29,30,  // s2f05 hf-
			 7,8,9,10,11,12,     // s2f05 hf+
			 19,20,21,22,23,24,  // s2f07 hf-
			 1,2,3,4,5,6};       // s2f07 hf+

   for( int iHLX = 0; iHLX < 36; ++iHLX ){
     HLXHFMap[iHLX] = HLXHFMapTemp[iHLX];
   }

   SetupHists();
   
   SetupEventInfo(iConfig);
}

HLXMonitor::~HLXMonitor()
{
  HLXTCP.Disconnect();
}

// ------------ Setup the monitoring elements ---------------
void
HLXMonitor::SetupHists()
{

  dbe_->setCurrentFolder(monitorName_+"/HFPlus");

   for( unsigned int iWedge = 0; iWedge < 18 && iWedge < NUM_HLX; ++iWedge )
   {  
      std::ostringstream tempStreamer;
      tempStreamer << std::dec << std::setw(2) << std::setfill('0') << (iWedge+1);

      std::ostringstream wedgeNum;      
      wedgeNum << std::dec << (iWedge % 18) + 1;

      dbe_->setCurrentFolder(monitorName_+"/HFPlus/Wedge"+tempStreamer.str());

      Set1Below[iWedge]   = dbe_->book1D("Set1_Below",   
				    "HF+ Wedge "+wedgeNum.str()+" Below Threshold 1 - Set 1",  
				    NBINS, XMIN, XMAX);
      Set1Between[iWedge] = dbe_->book1D("Set1_Between", 
				    "HF+ Wedge "+wedgeNum.str()+" Between Threshold 1 & 2 - Set 1",
				    NBINS, XMIN, XMAX);
      Set1Above[iWedge]   = dbe_->book1D("Set1_Above",   
				    "HF+ Wedge "+wedgeNum.str()+" Above Threshold 2 - Set 1",  
				    NBINS, XMIN, XMAX);
      Set2Below[iWedge]   = dbe_->book1D("Set2_Below",   
				    "HF+ Wedge "+wedgeNum.str()+" Below Threshold 1 - Set 2",  
				    NBINS, XMIN, XMAX);
      Set2Between[iWedge] = dbe_->book1D("Set2_Between", 
				    "HF+ Wedge "+wedgeNum.str()+" Between Threshold 1 & 2 - Set 2",
				    NBINS, XMIN, XMAX);
      Set2Above[iWedge]   = dbe_->book1D("Set2_Above",   
				    "HF+ Wedge "+wedgeNum.str()+" Above Threshold 2 - Set 2",  
				    NBINS, XMIN, XMAX);    
      ETSum[iWedge]       = dbe_->book1D("ETSum",        
				    "HF+ Wedge "+wedgeNum.str()+" E_{T} Sum",                
				    NBINS, XMIN, XMAX);    

      dbe_->tagContents(monitorName_+"/HFPlus/Wedge"+tempStreamer.str(), iWedge+1);
   }

   if(NUM_HLX > 17)
   {
      dbe_->setCurrentFolder(monitorName_+"/HFMinus");
    
      for( unsigned int iWedge=18; iWedge < NUM_HLX; ++iWedge )
      {
	 std::ostringstream tempStreamer;
	 tempStreamer << std::dec << std::setw(2) << std::setfill('0') << (iWedge+1);

	 std::ostringstream wedgeNum;      
	 wedgeNum << std::dec << (iWedge % 18) + 1;

	 dbe_->setCurrentFolder(monitorName_+"/HFMinus/Wedge"+tempStreamer.str());
	 Set1Below[iWedge]   = dbe_->book1D("Set1_Below",
				       "HF- Wedge "+wedgeNum.str()+" Below Threshold 1 - Set 1",  
				       NBINS, XMIN, XMAX);
	 Set1Between[iWedge] = dbe_->book1D("Set1_Between",
				       "HF- Wedge "+wedgeNum.str()+" Between Threshold 1 & 2 - Set 1",
				       NBINS, XMIN, XMAX);
	 Set1Above[iWedge]   = dbe_->book1D("Set1_Above",   
				       "HF- Wedge "+wedgeNum.str()+" Above Threshold 2 - Set 1",  
				       NBINS, XMIN, XMAX); 
	 Set2Below[iWedge]   = dbe_->book1D("Set2_Below",   
				       "HF- Wedge "+wedgeNum.str()+" Below Threshold 1 - Set 2",  
				       NBINS, XMIN, XMAX); 
	 Set2Between[iWedge] = dbe_->book1D("Set2_Between", 
				       "HF- Wedge "+wedgeNum.str()+" Between Threshold 1 & 2 - Set 2",
				       NBINS, XMIN, XMAX); 
	 Set2Above[iWedge]   = dbe_->book1D("Set2_Above",   
				       "HF- Wedge "+wedgeNum.str()+" Above Threshold 2 - Set 2",  
				       NBINS, XMIN, XMAX); 
	 ETSum[iWedge]       = dbe_->book1D("ETSum",        
				       "HF- Wedge "+wedgeNum.str()+" E_{T} Sum",                
				       NBINS, XMIN, XMAX); 

	 dbe_->tagContents(monitorName_+"/HFMinus/Wedge"+tempStreamer.str(), iWedge+1);
      }
   }

   if(!Accumulate){
     for( unsigned int iWedge = 0; iWedge < NUM_HLX; ++iWedge ){   
       Set1Below[iWedge]->  setResetMe(true);
       Set1Between[iWedge]->setResetMe(true);
       Set1Above[iWedge]->  setResetMe(true);
       Set2Below[iWedge]->  setResetMe(true);
       Set2Between[iWedge]->setResetMe(true);
       Set2Above[iWedge]->  setResetMe(true);
       ETSum[iWedge]->      setResetMe(true);    
     }
   }
  
   if(Style.compare("BX") == 0){
     OccXAxisTitle = "Bunch Crossing";
     OccYAxisTitle = "Tower Occupancy";
     EtXAxisTitle  = "Bunch Crossing";
     EtYAxisTitle  = "E_{T} Sum";
   }else if(Style.compare("Distribution")==0){
      OccXAxisTitle = "Tower Occupancy";
      OccYAxisTitle = "Count";
      EtXAxisTitle  = "E_{T} Sum";
      EtYAxisTitle  = "Count";
   }
//    else if(Style.compare("History")==0)
//    {
//       OccXAxisTitle = "Lumi Section";
//       OccYAxisTitle = "Avg Occupancy";
//       EtXAxisTitle  = "Lumi Section";
//       EtYAxisTitle  = "Avg Et Sum Occupancy";
    
//       Set1Below[HLXHFMap[i]]->setResetMe(false);
//       Set1Between[HLXHFMap[i]]->setResetMe(false);
//       Set1Above[HLXHFMap[i]]->setResetMe(false);
//       Set2Below[HLXHFMap[i]]->setResetMe(false);
//       Set2Between[HLXHFMap[i]]->setResetMe(false);
//       Set2Above[HLXHFMap[i]]->setResetMe(false);
//       ETSum[HLXHFMap[i]]->setResetMe(false);
    
//    }
  
   for( unsigned int iWedge=0; iWedge < NUM_HLX; ++iWedge )
   {
      Set1Below[iWedge]->  setAxisTitle(OccXAxisTitle, 1);
      Set1Below[iWedge]->  setAxisTitle(OccYAxisTitle, 2);
      Set1Between[iWedge]->setAxisTitle(OccXAxisTitle, 1);
      Set1Between[iWedge]->setAxisTitle(OccYAxisTitle, 2);
      Set1Above[iWedge]->  setAxisTitle(OccXAxisTitle, 1);
      Set1Above[iWedge]->  setAxisTitle(OccYAxisTitle, 2);	
      Set2Below[iWedge]->  setAxisTitle(OccXAxisTitle, 1);
      Set2Below[iWedge]->  setAxisTitle(OccYAxisTitle, 2);
      Set2Between[iWedge]->setAxisTitle(OccXAxisTitle, 1);
      Set2Between[iWedge]->setAxisTitle(OccYAxisTitle, 2);
      Set2Above[iWedge]->  setAxisTitle(OccXAxisTitle, 1);
      Set2Above[iWedge]->  setAxisTitle(OccYAxisTitle, 2);	
      ETSum[iWedge]->      setAxisTitle(EtXAxisTitle,  1);
      ETSum[iWedge]->      setAxisTitle(EtYAxisTitle,  2);	  
   }

   // Comparison Histograms
  
   dbe_->setCurrentFolder(monitorName_+"/HFCompare");

   std::string CompXTitle      = "HF Wedge";
   std::string CompEtSumYTitle = "E_{T} Sum per active tower";
   std::string CompOccYTitle   = "Occupancy per active tower";

   HFCompareEtSum = dbe_->book1D("HFCompareEtSum","E_{T} Sum - cyclic trigger ",NUM_HLX, 0, NUM_HLX );
   HFCompareEtSum->setAxisTitle( CompXTitle, 1 );
   HFCompareEtSum->setAxisTitle( CompEtSumYTitle, 2 );
 
   HFCompareOccBelowSet1 = dbe_->book1D("HFCompareOccBelowSet1",
					"Occupancy Below Threshold 1 - Set 1", 
					NUM_HLX, 0, NUM_HLX );
   HFCompareOccBelowSet1->setAxisTitle( CompXTitle, 1 );
   HFCompareOccBelowSet1->setAxisTitle( CompOccYTitle, 2 );

   HFCompareOccBetweenSet1 = dbe_->book1D("HFCompareOccBetweenSet1",
					  "Occupancy Between Threshold 1 & 2 - Set 1", 
					  NUM_HLX, 0, NUM_HLX );
   HFCompareOccBetweenSet1->setAxisTitle( CompXTitle, 1 );
   HFCompareOccBetweenSet1->setAxisTitle( CompOccYTitle, 2 );

   HFCompareOccAboveSet1 = dbe_->book1D("HFCompareOccAboveSet1",
					"Occupancy Above Threshold 2 - Set 1", 
					NUM_HLX, 0, NUM_HLX );
   HFCompareOccAboveSet1->setAxisTitle( CompXTitle, 1 );
   HFCompareOccAboveSet1->setAxisTitle( CompOccYTitle, 2 );

   HFCompareOccBelowSet2 = dbe_->book1D("HFCompareOccBelowSet2",
					"Occupancy Below Threshold 1 - Set 2", 
					NUM_HLX, 0, NUM_HLX);
   HFCompareOccBelowSet2->setAxisTitle(CompXTitle,1);
   HFCompareOccBelowSet2->setAxisTitle(CompOccYTitle,2);

   HFCompareOccBetweenSet2 = dbe_->book1D("HFCompareOccBetweenSet2",
					  "Occupancy Between Threshold 1 & 2 - Set 2", 
					  NUM_HLX, 0, NUM_HLX);
   HFCompareOccBetweenSet2->setAxisTitle(CompXTitle,1);
   HFCompareOccBetweenSet2->setAxisTitle(CompOccYTitle,2);

   HFCompareOccAboveSet2 = dbe_->book1D( "HFCompareOccAboveSet2",
					 "Occupancy Above Threshold 2 - Set 2", 
					 NUM_HLX, 0, NUM_HLX);
   HFCompareOccAboveSet2->setAxisTitle(CompXTitle,1);
   HFCompareOccAboveSet2->setAxisTitle(CompOccYTitle,2);

   // Average Histograms

   dbe_->setCurrentFolder(monitorName_+"/Average");

   int    OccBins = 10000;  // This does absolutely nothing. 
   double OccMin  = 0; 
   double OccMax  = 0; // If min and max are zero, no bounds on the data are set.

   int    EtSumBins = 10000; // This does absolutely nothing.  The Variable is not used in the function.
   double EtSumMin  = 0;
   double EtSumMax  = 0;  // If min and max are zero, no bounds on the data are set.

   std::string errorOpt = "i"; 
   
   std::string AvgXTitle      = "HF Wedge";
   std::string AvgEtSumYTitle = "Average E_{T} Sum";
   std::string AvgOccYTitle   = "Average Tower Occupancy";

   AvgEtSum        = dbe_->bookProfile( "AvgEtSum", 
					"Average E_{T} Sum",          
					NUM_HLX, 0, NUM_HLX, EtSumBins, EtSumMin, EtSumMax);
   AvgEtSum->setAxisTitle( AvgXTitle, 1 );
   AvgEtSum->setAxisTitle( AvgEtSumYTitle, 2 );

   AvgOccBelowSet1 = dbe_->bookProfile( "AvgOccBelowSet1",     
					"Average Occupancy Below Threshold 1 - Set1",   
					NUM_HLX, 0, NUM_HLX, OccBins, OccMin, OccMax, errorOpt.c_str());
   AvgOccBelowSet1->setAxisTitle( AvgXTitle, 1 );
   AvgOccBelowSet1->setAxisTitle( AvgOccYTitle, 2 );

   AvgOccBetweenSet1 = dbe_->bookProfile( "AvgOccBetweenSet1", 
					  "Average Occupancy Between Threhold 1 & 2 - Set1", 
					  NUM_HLX, 0, NUM_HLX, OccBins, OccMin, OccMax, errorOpt.c_str());
   AvgOccBetweenSet1->setAxisTitle( AvgXTitle, 1 );
   AvgOccBetweenSet1->setAxisTitle( AvgOccYTitle, 2 );

   AvgOccAboveSet1 = dbe_->bookProfile( "AvgOccAboveSet1",
					"Average Occupancy Above Threshold 2 - Set1",
					  NUM_HLX, 0, NUM_HLX, OccBins, OccMin, OccMax, errorOpt.c_str());
   AvgOccAboveSet1->setAxisTitle( AvgXTitle, 1 );
   AvgOccAboveSet1->setAxisTitle( AvgOccYTitle, 2 );

   AvgOccBelowSet2 = dbe_->bookProfile("AvgOccBelowSet2",
				       "Average Occupancy Below Threshold 1 - Set2",
					  NUM_HLX, 0, NUM_HLX, OccBins, OccMin, OccMax, errorOpt.c_str());
   AvgOccBelowSet2->setAxisTitle( AvgXTitle, 1 );
   AvgOccBelowSet2->setAxisTitle( AvgOccYTitle, 2 );

   AvgOccBetweenSet2 = dbe_->bookProfile("AvgOccBetweenSet2",
					 "Average Occupancy Between Threshold 1 & 2 - Set2",
					 NUM_HLX, 0, NUM_HLX, OccBins, OccMin, OccMax, errorOpt.c_str());
   AvgOccBetweenSet2->setAxisTitle( AvgXTitle, 1 );
   AvgOccBetweenSet2->setAxisTitle( AvgOccYTitle, 2 );

   AvgOccAboveSet2 = dbe_->bookProfile("AvgOccAboveSet2",
				       "Average Occupancy Above Threshold 2 - Set2",
				       NUM_HLX, 0, NUM_HLX, OccBins, OccMin, OccMax, errorOpt.c_str());
   AvgOccAboveSet2->setAxisTitle( AvgXTitle, 1 );
   AvgOccAboveSet2->setAxisTitle( AvgOccYTitle, 2 );


   // Luminosity Histograms
   dbe_->setCurrentFolder(monitorName_+"/Luminosity");

   std::string LumiXTitle      = "Bunch Crossing";
   std::string LumiEtSumYTitle = "Luminosity: E_{T} Sum";
   std::string LumiOccYTitle   = "Luminosity: Occupancy";

   LumiEtSum = dbe_->bookProfile("LumiEtSum","Luminosity ",NBINS, XMIN, XMAX, EtSumBins, EtSumMin, EtSumMax );
   LumiEtSum->setAxisTitle( LumiXTitle, 1 );
   LumiEtSum->setAxisTitle( LumiEtSumYTitle, 2 );
 
   LumiOccSet1 = dbe_->bookProfile("LumiOccSet1","Luminosity - Set 1", NBINS, XMIN, XMAX, OccBins, OccMax, OccMin );
   LumiOccSet1->setAxisTitle( LumiXTitle, 1 );
   LumiOccSet1->setAxisTitle( LumiOccYTitle, 2 );

   LumiOccSet2 = dbe_->bookProfile("LumiOccSet2","Luminosity - Set 2", NBINS, XMIN, XMAX, OccBins, OccMax, OccMin );
   LumiOccSet2->setAxisTitle( LumiXTitle, 1 );
   LumiOccSet2->setAxisTitle( LumiOccYTitle, 2 );

   LumiDiffEtSumOcc1 = dbe_->bookProfile("LumiDiffEtSumOcc1","Luminosity ",NBINS, XMIN, XMAX, OccBins, OccMax, OccMin  );
   LumiDiffEtSumOcc1->setAxisTitle( LumiXTitle, 1 );
   LumiDiffEtSumOcc1->setAxisTitle( LumiEtSumYTitle, 2 );

   LumiDiffEtSumOcc2 = dbe_->bookProfile("LumiDiffEtSumOcc2","Luminosity ",NBINS, XMIN, XMAX, OccBins, OccMax, OccMin  );
   LumiDiffEtSumOcc2->setAxisTitle( LumiXTitle, 1 );
   LumiDiffEtSumOcc2->setAxisTitle( LumiEtSumYTitle, 2 );

   LumiDiffOcc1Occ2 = dbe_->bookProfile("LumiDiffOcc1Occ2","Luminosity ",NBINS, XMIN, XMAX, OccBins, OccMax, OccMin  );
   LumiDiffOcc1Occ2->setAxisTitle( LumiXTitle, 1 );
   LumiDiffOcc1Occ2->setAxisTitle( LumiEtSumYTitle, 2 );

   // Sanity check sum histograms
   dbe_->setCurrentFolder(monitorName_+"/CheckSums");

   std::string sumXTitle   = "HF Wedge";
   std::string sumYTitle   = "Occupancy Sum (Below+Above+Between)";

   SumAllOccSet1 = dbe_->bookProfile("SumAllOccSet1","Occupancy Check - Set 1",NUM_HLX, 0, NUM_HLX, OccBins, OccMax, OccMin );
   SumAllOccSet1->setAxisTitle( sumXTitle, 1 );
   SumAllOccSet1->setAxisTitle( sumYTitle, 2 );

   SumAllOccSet2 = dbe_->bookProfile("SumAllOccSet2","Occupancy Check - Set 2",NUM_HLX, 0, NUM_HLX, OccBins, OccMax, OccMin );
   SumAllOccSet2->setAxisTitle( sumXTitle, 1 );
   SumAllOccSet2->setAxisTitle( sumYTitle, 2 );
 
   dbe_->showDirStructure();
}


void HLXMonitor::SetupEventInfo(const edm::ParameterSet & ps)
{

  using std::string;

  parameters_ = ps;
  pEvent_ = 0;
  evtRateCount_ = 0;
  gettimeofday(&currentTime_,NULL);
  lastAvgTime_ = currentTime_;
  
  dbe_ = edm::Service<DQMStore>().operator->();

  string eventinfofolder = parameters_.getUntrackedParameter<std::string>("eventInfoFolder", "EventInfo") ;
  string subsystemname = parameters_.getUntrackedParameter<std::string>("subSystemFolder", "YourSubsystem") ;
  string currentfolder = subsystemname + "/" +  eventinfofolder ;
  cout << "currentfolder " << currentfolder << endl;

  evtRateWindow_ = parameters_.getUntrackedParameter<double>("eventRateWindow", 0.5);
  if(evtRateWindow_<=0.15) evtRateWindow_=0.15;
  cout << "Event Rate averaged over " << evtRateWindow_ << " minutes" << endl;

  dbe_->setCurrentFolder(currentfolder) ;

  //Event specific contents
  runId_     = dbe_->bookInt("iRun");
  lumisecId_ = dbe_->bookInt("iLumiSection");
}


// ------------ method called to for each event  ------------
void
HLXMonitor::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   int errorCode = 0;
   do
   {
      errorCode = HLXTCP.ReceiveLumiSection(lumiSection);
      cout << "ReceiveLumiSection: " << errorCode << endl;

      while(errorCode !=1)
      {
	 HLXTCP.Disconnect();
	 cout << "Connecting to TCPDistributor" << endl;
	 errorCode = HLXTCP.Connect();
	 if(errorCode != 1)
	 {
	    cout << "*** Connection Failed: " << errorCode 
		 << " Attempting reconnect in " << reconnTime << " seconds." << endl;
	    sleep(reconnTime);
	 }
      }    
   } while( errorCode != 1 );
  
   // Fill the monitoring histograms 
   if(Style.compare("BX") == 0)
   {
      FillHistoBX(lumiSection);
   }
   else if(Style.compare("Dist")==0)
   {
      FillHistoDist(lumiSection);
   }
   // not implemented
   //else if(Style.compare("History")==0)
   //{
      // FillHistoHistory(lumiSection);
   //}

   FillHistoHFCompare(lumiSection);
   FillHistoAvg(lumiSection);
   FillHistoLumi(lumiSection);
   FillHistoSum(lumiSection);
   FillEventInfo(lumiSection);

   // Not implemented
//    if(lumiSection.hdr.runNumber != runNumber_)
//    {
//       //SaveDQMFile();  
//       runNumber_ = lumiSection.hdr.runNumber;
//       if(ResetAtNewRun)
//       {
// 	 for( unsigned int i=0; i < NUM_HLX; ++i )
// 	 {
// 	    //need a good way to do this
// 	    Set1Below[  HLXHFMap[i]]->softReset();
// 	    Set1Between[HLXHFMap[i]]->softReset();
// 	    Set1Above[  HLXHFMap[i]]->softReset();
// 	    Set2Below[  HLXHFMap[i]]->softReset();
// 	    Set2Between[HLXHFMap[i]]->softReset();
// 	    Set2Above[  HLXHFMap[i]]->softReset();
// 	    ETSum[      HLXHFMap[i]]->softReset();    
// 	 }
//       }
//    }

   cout << "Run: " << lumiSection.hdr.runNumber 
	<< " Section: " << lumiSection.hdr.sectionNumber 
	<< " Orbit: " << lumiSection.hdr.startOrbit << endl;
   cout << "Et Lumi: " << lumiSection.lumiSummary.InstantETLumi << endl;
   cout << "Occ Lumi 1: " << lumiSection.lumiSummary.InstantOccLumi[0] << endl;
   cout << "Occ Lumi 2: " << lumiSection.lumiSummary.InstantOccLumi[1] << endl;
   cout << "Noise[0]: " << lumiSection.lumiSummary.lumiNoise[0] << endl;
   cout << "Noise[1]: " << lumiSection.lumiSummary.lumiNoise[1] << endl;

//    if( (lumiSection.hdr.sectionNumber % SavePeriod == 0) && (SavePeriod != -1))
//    {
//       SaveDQMFile();
//    }
}


void HLXMonitor::SaveDQMFile(){

  std::ostringstream tempStreamer;
  tempStreamer << OutputDir << "/" << OutputFilePrefix 
	       << "_" << std::setfill('0') << std::setw(runNumLength) 
	       << runNumber_ 
	       << "_" << std::setfill('0') << std::setw(secNumLength) 
	       << lumiSection.hdr.sectionNumber
	       << ".root";
  dbe_->save(tempStreamer.str());
}

// ------------ method called once each job just before starting event loop  ------------
void HLXMonitor::beginJob(const edm::EventSetup&)
{ 
   HLXTCP.SetIP(DistribIP);

   int errorCode = HLXTCP.SetPort(listenPort);
   cout << "SetPort: " << errorCode << endl;
   errorCode = HLXTCP.SetMode(AquireMode);
   cout << "AquireMode: " << errorCode << endl;
  
   do
   {
      // cout << "Connecting to TCPDistributor" << endl;
      errorCode = HLXTCP.Connect();
      if(errorCode != 1)
      {
	 cout << "Attempting to reconnect in " << reconnTime << " seconds." << endl;
	 sleep(reconnTime);
      }
   } while(errorCode != 1);
}

// ------------ method called once each job just after ending the event loop  ------------
void HLXMonitor::endJob() 
{
   HLXTCP.Disconnect();
}

void HLXMonitor::FillHistoAvg(const LUMI_SECTION & section)
{

   for( unsigned int iHLX = 0; iHLX < NUM_HLX; ++iHLX )
   {
      if(section.occupancy[iHLX].hdr.numNibbles != 0)
      {
	for( unsigned int iBX = 0; iBX < (NUM_BUNCHES - 100); ++iBX )  // Don't include the last one hundred BX in the average.
	 {
	   
	   unsigned int iWedge = HLXHFMap[iHLX];
	   
	   AvgEtSum->Fill( iWedge,section.etSum[iHLX].data[iBX]);
	
	    AvgOccBelowSet1->  Fill( iWedge, section.occupancy[iHLX].data[set1BelowIndex  ][iBX] );
	    AvgOccBetweenSet1->Fill( iWedge, section.occupancy[iHLX].data[set1BetweenIndex][iBX] );
	    AvgOccAboveSet1->  Fill( iWedge, section.occupancy[iHLX].data[set1AboveIndex  ][iBX] );
	   
	    AvgOccBelowSet2->  Fill( iWedge, section.occupancy[iHLX].data[set2BelowIndex  ][iBX] );
	    AvgOccBetweenSet2->Fill( iWedge, section.occupancy[iHLX].data[set2BetweenIndex][iBX] );
	    AvgOccAboveSet2->  Fill( iWedge, section.occupancy[iHLX].data[set2AboveIndex  ][iBX] );
	
	 }
      }
   }
}

void HLXMonitor::FillHistoBX(const LUMI_SECTION & section)
{

   for( unsigned int iHLX = 0; iHLX < NUM_HLX; ++iHLX )
   {
      if(section.occupancy[iHLX].hdr.numNibbles != 0)
      {
	 for( unsigned int iBX = 0; iBX < NUM_BUNCHES; ++iBX)
	 { 

	   unsigned int iWedge = HLXHFMap[iHLX];

	    Set1Below[iWedge]->  Fill(iBX, section.occupancy[iHLX].data[set1BelowIndex  ][iBX]);
	    Set1Between[iWedge]->Fill(iBX, section.occupancy[iHLX].data[set1BetweenIndex][iBX]);
	    Set1Above[iWedge]->  Fill(iBX, section.occupancy[iHLX].data[set1AboveIndex  ][iBX]);
	    Set2Below[iWedge]->  Fill(iBX, section.occupancy[iHLX].data[set2BelowIndex  ][iBX]);
	    Set2Between[iWedge]->Fill(iBX, section.occupancy[iHLX].data[set2BetweenIndex][iBX]);
	    Set2Above[iWedge]->  Fill(iBX, section.occupancy[iHLX].data[set2AboveIndex  ][iBX]);
	    ETSum[iWedge]->      Fill(iBX, section.etSum[iHLX].data[iBX]);
	 }
      }
   }
}

void HLXMonitor::FillHistoLumi(const LUMI_SECTION & section)
{

   for( unsigned int iHLX = 0; iHLX < NUM_HLX; ++iHLX )
   {
      if(section.occupancy[iHLX].hdr.numNibbles != 0)
      {
	 for( unsigned int iBX = 0; iBX < NUM_BUNCHES; ++iBX)
	 { 
	    LumiEtSum->Fill(iBX, section.lumiDetail.ETLumi[iBX]);
	    LumiOccSet1->Fill(iBX, section.lumiDetail.OccLumi[0][iBX]);
	    LumiOccSet2->Fill(iBX, section.lumiDetail.OccLumi[1][iBX]);

	    LumiDiffEtSumOcc1->Fill(iBX, (section.lumiDetail.ETLumi[iBX]-section.lumiDetail.OccLumi[0][iBX]));
	    LumiDiffEtSumOcc2->Fill(iBX, (section.lumiDetail.ETLumi[iBX]-section.lumiDetail.OccLumi[1][iBX]));
	    LumiDiffOcc1Occ2->Fill(iBX, (section.lumiDetail.ETLumi[iBX]-section.lumiDetail.OccLumi[1][iBX]));
	 }
      }
   }
}


void HLXMonitor::FillHistoSum(const LUMI_SECTION & section)
{

   for( unsigned int iHLX = 0; iHLX < NUM_HLX; ++iHLX )
   {
      if(section.occupancy[iHLX].hdr.numNibbles != 0)
      {
	 unsigned int iWedge = HLXHFMap[iHLX];  
	 float total1 = 0;
	 float total2 = 0;
	 for( unsigned int iBX = 0; iBX < (NUM_BUNCHES - 100); ++iBX )  // Don't include the last one hundred BX in the average.
	 //for( unsigned int iBX = 50; iBX < 51; ++iBX )  // Don't include the last one hundred BX in the average.
	 {
	    total1 += (float)section.occupancy[iHLX].data[set1BelowIndex  ][iBX];
	    total1 += (float)section.occupancy[iHLX].data[set1BetweenIndex][iBX];
	    total1 += (float)section.occupancy[iHLX].data[set1AboveIndex  ][iBX];

	    total2 += (float)section.occupancy[iHLX].data[set2BelowIndex  ][iBX];
	    total2 += (float)section.occupancy[iHLX].data[set2BetweenIndex][iBX];
	    total2 += (float)section.occupancy[iHLX].data[set2AboveIndex  ][iBX];
	 }

	 //cout << "At HLX " << iHLX << " " << total1 << " " << total2 << endl;
	 total1 = total1/(float)(NUM_BUNCHES-100);
	 total2 = total2/(float)(NUM_BUNCHES-100);
	 //cout << "At HLX " << iHLX << " " << total1 << " " << total2 << endl << endl;

	 SumAllOccSet1->  Fill( iWedge, total1 );
	 SumAllOccSet2->  Fill( iWedge, total2 );
      }
   }
}


void HLXMonitor::FillHistoDist(const LUMI_SECTION & section)
{
   for( unsigned int iHLX = 0; iHLX < NUM_HLX; ++iHLX )
   {
      if(section.occupancy[iHLX].hdr.numNibbles != 0)
      {
	 for( unsigned int iBX = 0; iBX < NUM_BUNCHES; ++iBX )
	 { 

	   unsigned int iWedge = HLXHFMap[iHLX];

	    Set1Below[iWedge]->  Fill( section.occupancy[iHLX].data[set1BelowIndex  ][iBX] );
	    Set1Between[iWedge]->Fill( section.occupancy[iHLX].data[set1BetweenIndex][iBX] );
	    Set1Above[iWedge]->  Fill( section.occupancy[iHLX].data[set1AboveIndex  ][iBX] );
	    Set2Below[iWedge]->  Fill( section.occupancy[iHLX].data[set2BelowIndex  ][iBX] );
	    Set2Between[iWedge]->Fill( section.occupancy[iHLX].data[set2BetweenIndex][iBX] );
	    Set2Above[iWedge]->  Fill( section.occupancy[iHLX].data[set2AboveIndex  ][iBX] );
	    ETSum[iWedge]->      Fill( section.etSum[iHLX].data[iBX] );
	 }
      }
   }
}

void HLXMonitor::FillHistoHFCompare(const LUMI_SECTION & section)
{

  for( unsigned int iHLX = 0; iHLX < NUM_HLX; ++iHLX ){

    unsigned int iWedge = HLXHFMap[iHLX];
      
    if(section.occupancy[iHLX].hdr.numNibbles != 0){
      float nActvTwrsSet1 = section.occupancy[iHLX].data[set1AboveIndex][TriggerBX]
	+ section.occupancy[iHLX].data[set1BetweenIndex][TriggerBX]
	+ section.occupancy[iHLX].data[set1BelowIndex][TriggerBX];
      
      float nActvTwrsSet2 = section.occupancy[iHLX].data[set2AboveIndex][TriggerBX]
	+ section.occupancy[iHLX].data[set2BetweenIndex][TriggerBX]
	+ section.occupancy[iHLX].data[set2BelowIndex][TriggerBX];
      
      float total = nActvTwrsSet1 + nActvTwrsSet2;
      
      if( total > 0){  
	float tempData = section.etSum[iHLX].data[TriggerBX]/total;
	//cout << "Filling HFCompare Et sum " << tempData << endl;
	HFCompareEtSum->Fill( iWedge, tempData );
      }
      
      if(nActvTwrsSet1 > 0){
	float tempData = (float)section.occupancy[iHLX].data[set1BelowIndex][TriggerBX]/nActvTwrsSet1;
	HFCompareOccBelowSet1->Fill( iWedge, tempData);
	
	tempData = (float)section.occupancy[iHLX].data[set1BetweenIndex][TriggerBX]/nActvTwrsSet1;
	HFCompareOccBetweenSet1->Fill( iWedge, tempData); 
	
	tempData = (float)section.occupancy[iHLX].data[set1AboveIndex][TriggerBX]/nActvTwrsSet1;
	HFCompareOccAboveSet1->Fill( iWedge, tempData); 
      }
      
      if( nActvTwrsSet2 > 0){
	float tempData = (float)section.occupancy[iHLX].data[set2BelowIndex][TriggerBX]/nActvTwrsSet2;
	HFCompareOccBelowSet2->Fill( iWedge, tempData);
	
	tempData = (float)section.occupancy[iHLX].data[set2BetweenIndex][TriggerBX]/nActvTwrsSet2;
	HFCompareOccBetweenSet2->Fill( iWedge, tempData); 
	
	tempData = (float)section.occupancy[iHLX].data[set2AboveIndex][TriggerBX]/nActvTwrsSet2;
	HFCompareOccAboveSet2->Fill( iWedge, tempData); 
      }
    }
  }
}

void HLXMonitor::FillEventInfo(const LUMI_SECTION & section)
{
   runId_->Fill( section.hdr.runNumber );
   lumisecId_->Fill( (int)(section.hdr.sectionNumber/64) + 1 );
}

// void HLXMonitor::FillHistoHistory(const LUMI_SECTION & section)
// {
//   unsigned int i;
//   float ETSumData;
//   float avgOcc[6];

//   for(i=0; i<NUM_HLX; i++){
    
//     AvgOccupancy(section.occupancy[iHLX], avgOcc, NUM_BUNCHES,1);
//     ETSumData = AvgETSum(section.etSum[iHLX],NUM_BUNCHES,1);

//     if(Accumulate==true){
//       avgOcc[set1BelowIndex]   += Set1Below[iWedge]->getBinContent(counter-1);
//       avgOcc[set1BetweenIndex] += Set1Between[iWedge]->getBinContent(counter-1);
//       avgOcc[set1AboveIndex]   += Set1Above[iWedge]->getBinContent(counter-1);
//       avgOcc[set2BelowIndex]   += Set2Below[iWedge]->getBinContent(counter-1);
//       avgOcc[set2BetweenIndex] += Set2Between[iWedge]->getBinContent(counter-1);
//       avgOcc[set2AboveIndex]   += Set2Above[iWedge]->getBinContent(counter-1);
//       ETSumData                += ETSum[iWedge]->getBinContent(counter-1);
//     }

//     Set1Below[iWedge]   ->Fill(counter,avgOcc[set1BelowIndex]);
//     Set1Between[iWedge] ->Fill(counter,avgOcc[set1BetweenIndex]);
//     Set1Above[iWedge]   ->Fill(counter,avgOcc[set1AboveIndex]);
//     Set2Below[iWedge]   ->Fill(counter,avgOcc[set2BelowIndex]);
//     Set2Between[iWedge] ->Fill(counter,avgOcc[set2BetweenIndex]);
//     Set2Above[iWedge]   ->Fill(counter,avgOcc[set2AboveIndex]);
//     ETSum[iWedge]       ->Fill(counter,ETSumData);

//     //setAxisRange doesn't seem to do anything yet.  
//     Set1Below[iWedge]  ->setAxisRange(0,counter+1,1);
//     Set1Between[iWedge]->setAxisRange(0,counter+1,1);
//     Set1Above[iWedge]  ->setAxisRange(0,counter+1,1);
//     Set2Below[iWedge]  ->setAxisRange(0,counter+1,1);
//     Set2Between[iWedge]->setAxisRange(0,counter+1,1);
//     Set2Above[iWedge]  ->setAxisRange(0,counter+1,1);
//     ETSum[iWedge]      ->setAxisRange(0,counter+1,1);
//   }
// }

//define this as a plug-in

DEFINE_FWK_MODULE(HLXMonitor);
