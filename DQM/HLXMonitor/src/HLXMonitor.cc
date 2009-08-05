/*
    Author:  Adam Hunt
    email:   ahunt@princeton.edu
*/
#include "DQM/HLXMonitor/interface/HLXMonitor.h"

// STL Headers

#include <math.h>
#include <iomanip>
#include <TSystem.h>

using std::cout;
using std::endl;

HLXMonitor::HLXMonitor(const edm::ParameterSet& iConfig)
{

   NUM_HLX          = iConfig.getUntrackedParameter< unsigned int >("numHlx",     36);
   NUM_BUNCHES      = iConfig.getUntrackedParameter< unsigned int >("numBunches", 3564);
   MAX_LS           = iConfig.getUntrackedParameter< unsigned int >("maximumNumLS", 480);
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
   DistribIP1       = iConfig.getUntrackedParameter< std::string  >("PrimaryHLXDAQIP", "vmepcs2f17-18");
   DistribIP2       = iConfig.getUntrackedParameter< std::string  >("SecondaryHLXDAQIP", "vmepcs2f17-19");
   ResetAtNewRun    = iConfig.getUntrackedParameter< bool         >("NewRun_Reset","true");
   SaveAtEndJob     = iConfig.getUntrackedParameter< bool         >("SaveAtEndJob","true");

   eventInfoFolder_ = iConfig.getUntrackedParameter<std::string   >("eventInfoFolder", "EventInfo") ;
   subSystemName_   = iConfig.getUntrackedParameter<std::string   >("subSystemName", "HLX") ;

   // Set the lumi section counter
   lsBinOld = 0;
   lumiSectionCount = 0;
   sectionInstantSumEt = 0;
   sectionInstantErrSumEt = 0;
   sectionInstantSumOcc1 = 0;
   sectionInstantErrSumOcc1 = 0;
   sectionInstantSumOcc2 = 0;
   sectionInstantErrSumOcc2 = 0;
   sectionInstantNorm = 0;

   // HLX Config info
   set1BelowIndex   = 0;
   set1BetweenIndex = 1;
   set1AboveIndex   = 2;
   set2BelowIndex   = 3;
   set2BetweenIndex = 4;
   set2AboveIndex   = 5;

   runNumLength     = 9;
   secNumLength     = 8;

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
   //cout << "Monitor name = " << monitorName_ << endl;
   prescaleEvt_ = iConfig.getUntrackedParameter<int>("prescaleEvt", -1);
   //cout << "===>DQM event prescale = " << prescaleEvt_ << " events "<< endl;

   unsigned int HLXHFMapTemp[] = {31,32,33,34,35,18,  // s2f07 hf-
				  13,14,15,16,17,0,   // s2f07 hf+
				  25,26,27,28,29,30,  // s2f05 hf-
				  7, 8, 9, 10,11,12,     // s2f05 hf+
				  19,20,21,22,23,24,  // s2f02 hf-
				  1, 2, 3, 4, 5, 6};       // s2f02 hf+

   runNumber_       = 0;
   expectedNibbles_ = 0;

   for( int iHLX = 0; iHLX < 36; ++iHLX ){
     HLXHFMap[iHLX] = HLXHFMapTemp[iHLX];
     //std::cout << "At " << iHLX << " Wedge " << HLXHFMap[iHLX] << std::endl;
     totalNibbles_[iHLX] = 0;
   }

   SetupHists();
   SetupEventInfo();
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

   LumiAvgEtSum = dbe_->bookProfile("LumiAvgEtSum","Average Luminosity ",int(XMAX-XMIN), XMIN, XMAX, EtSumBins, EtSumMin, EtSumMax );
   LumiAvgEtSum->setAxisTitle( LumiXTitle, 1 );
   LumiAvgEtSum->setAxisTitle( LumiEtSumYTitle, 2 );
 
   LumiAvgOccSet1 = dbe_->bookProfile("LumiAvgOccSet1","Average Luminosity - Set 1", int(XMAX-XMIN), XMIN, XMAX, OccBins, OccMax, OccMin );
   LumiAvgOccSet1->setAxisTitle( LumiXTitle, 1 );
   LumiAvgOccSet1->setAxisTitle( LumiOccYTitle, 2 );

   LumiAvgOccSet2 = dbe_->bookProfile("LumiAvgOccSet2","Average Luminosity - Set 2", int(XMAX-XMIN), XMIN, XMAX, OccBins, OccMax, OccMin );
   LumiAvgOccSet2->setAxisTitle( LumiXTitle, 1 );
   LumiAvgOccSet2->setAxisTitle( LumiOccYTitle, 2 );

   LumiInstantEtSum = dbe_->book1D("LumiInstantEtSum","Instantaneous Luminosity ",int(XMAX-XMIN), XMIN, XMAX );
   LumiInstantEtSum->setAxisTitle( LumiXTitle, 1 );
   LumiInstantEtSum->setAxisTitle( LumiEtSumYTitle, 2 );
 
   LumiInstantOccSet1 = dbe_->book1D("LumiInstantOccSet1","Instantaneous Luminosity - Set 1", int(XMAX-XMIN), XMIN, XMAX );
   LumiInstantOccSet1->setAxisTitle( LumiXTitle, 1 );
   LumiInstantOccSet1->setAxisTitle( LumiOccYTitle, 2 );

   LumiInstantOccSet2 = dbe_->book1D("LumiInstantOccSet2","Instantaneous Luminosity - Set 2", int(XMAX-XMIN), XMIN, XMAX );
   LumiInstantOccSet2->setAxisTitle( LumiXTitle, 1 );
   LumiInstantOccSet2->setAxisTitle( LumiOccYTitle, 2 );

   LumiIntegratedEtSum = dbe_->book1D("LumiIntegratedEtSum","Integrated Luminosity ",int(XMAX-XMIN), XMIN, XMAX );
   LumiIntegratedEtSum->setAxisTitle( LumiXTitle, 1 );
   LumiIntegratedEtSum->setAxisTitle( LumiEtSumYTitle, 2 );
 
   LumiIntegratedOccSet1 = dbe_->book1D("LumiIntegratedOccSet1","Integrated Luminosity - Set 1", int(XMAX-XMIN), XMIN, XMAX );
   LumiIntegratedOccSet1->setAxisTitle( LumiXTitle, 1 );
   LumiIntegratedOccSet1->setAxisTitle( LumiOccYTitle, 2 );

   LumiIntegratedOccSet2 = dbe_->book1D("LumiIntegratedOccSet2","Integrated Luminosity - Set 2", int(XMAX-XMIN), XMIN, XMAX );
   LumiIntegratedOccSet2->setAxisTitle( LumiXTitle, 1 );
   LumiIntegratedOccSet2->setAxisTitle( LumiOccYTitle, 2 );


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

   // History histograms
   dbe_->setCurrentFolder(monitorName_+"/HistoryRaw");

   std::string HistXTitle = "Time (LS)";
   std::string RecentHistXTitle = "Time (LS/64)";
   std::string HistEtSumYTitle = "Average E_{T} Sum";
   std::string HistOccYTitle = "Average Occupancy";
   std::string HistLumiYTitle = "Luminosity";
   std::string BXvsTimeXTitle = "Time (LS)";
   std::string BXvsTimeYTitle = "BX";

   // Et Sum histories
   HistAvgEtSumHFP        = dbe_->bookProfile( "HistAvgEtSumHFP", "Average Et Sum: HF+",          
   MAX_LS, 0.5, (double)MAX_LS+0.5, EtSumBins, EtSumMin, EtSumMax);
   HistAvgEtSumHFP->setAxisTitle( HistXTitle, 1 );
   HistAvgEtSumHFP->setAxisTitle( HistEtSumYTitle, 2 );

   HistAvgEtSumHFM        = dbe_->bookProfile( "HistAvgEtSumHFM", "Average Et Sum: HF-",          
   MAX_LS, 0.5, (double)MAX_LS+0.5, EtSumBins, EtSumMin, EtSumMax);
   HistAvgEtSumHFM->setAxisTitle( HistXTitle, 1 );
   HistAvgEtSumHFM->setAxisTitle( HistEtSumYTitle, 2 );

   // Tower Occupancy Histories
   HistAvgOccBelowSet1HFP = dbe_->bookProfile( "HistAvgOccBelowSet1HFP", "Average Occ Set1Below: HF+",
   MAX_LS, 0.5, (double)MAX_LS+0.5, OccBins, OccMin, OccMax );
   HistAvgOccBelowSet1HFP->setAxisTitle( HistXTitle, 1 );
   HistAvgOccBelowSet1HFP->setAxisTitle( HistOccYTitle, 2 );

   HistAvgOccBelowSet1HFM = dbe_->bookProfile( "HistAvgOccBelowSet1HFM", "Average Occ Set1Below: HF-",
   MAX_LS, 0.5, (double)MAX_LS+0.5, OccBins, OccMin, OccMax );
   HistAvgOccBelowSet1HFM->setAxisTitle( HistXTitle, 1 );
   HistAvgOccBelowSet1HFM->setAxisTitle( HistOccYTitle, 2 );

   HistAvgOccBetweenSet1HFP = dbe_->bookProfile( "HistAvgOccBetweenSet1HFP", "Average Occ Set1Between: HF+",
   MAX_LS, 0.5, (double)MAX_LS+0.5, OccBins, OccMin, OccMax );
   HistAvgOccBetweenSet1HFP->setAxisTitle( HistXTitle, 1 );
   HistAvgOccBetweenSet1HFP->setAxisTitle( HistOccYTitle, 2 );

   HistAvgOccBetweenSet1HFM = dbe_->bookProfile( "HistAvgOccBetweenSet1HFM", "Average Occ Set1Between: HF-",
   MAX_LS, 0.5, (double)MAX_LS+0.5, OccBins, OccMin, OccMax );
   HistAvgOccBetweenSet1HFM->setAxisTitle( HistXTitle, 1 );
   HistAvgOccBetweenSet1HFM->setAxisTitle( HistOccYTitle, 2 );

   HistAvgOccAboveSet1HFP = dbe_->bookProfile( "HistAvgOccAboveSet1HFP", "Average Occ Set1Above: HF+",
   MAX_LS, 0.5, (double)MAX_LS+0.5, OccBins, OccMin, OccMax );
   HistAvgOccAboveSet1HFP->setAxisTitle( HistXTitle, 1 );
   HistAvgOccAboveSet1HFP->setAxisTitle( HistOccYTitle, 2 );

   HistAvgOccAboveSet1HFM = dbe_->bookProfile( "HistAvgOccAboveSet1HFM", "Average Occ Set1Above: HF-",
   MAX_LS, 0.5, (double)MAX_LS+0.5, OccBins, OccMin, OccMax );
   HistAvgOccAboveSet1HFM->setAxisTitle( HistXTitle, 1 );
   HistAvgOccAboveSet1HFM->setAxisTitle( HistOccYTitle, 2 );

   HistAvgOccBelowSet2HFP = dbe_->bookProfile( "HistAvgOccBelowSet2HFP", "Average Occ Set2Below: HF+",
   MAX_LS, 0.5, (double)MAX_LS+0.5, OccBins, OccMin, OccMax );
   HistAvgOccBelowSet2HFP->setAxisTitle( HistXTitle, 1 );
   HistAvgOccBelowSet2HFP->setAxisTitle( HistOccYTitle, 2 );

   HistAvgOccBelowSet2HFM = dbe_->bookProfile( "HistAvgOccBelowSet2HFM", "Average Occ Set2Below: HF-",
   MAX_LS, 0.5, (double)MAX_LS+0.5, OccBins, OccMin, OccMax );
   HistAvgOccBelowSet2HFM->setAxisTitle( HistXTitle, 1 );
   HistAvgOccBelowSet2HFM->setAxisTitle( HistOccYTitle, 2 );

   HistAvgOccBetweenSet2HFP = dbe_->bookProfile( "HistAvgOccBetweenSet2HFP", "Average Occ Set2Between: HF+",
   MAX_LS, 0.5, (double)MAX_LS+0.5, OccBins, OccMin, OccMax );
   HistAvgOccBetweenSet2HFP->setAxisTitle( HistXTitle, 1 );
   HistAvgOccBetweenSet2HFP->setAxisTitle( HistOccYTitle, 2 );

   HistAvgOccBetweenSet2HFM = dbe_->bookProfile( "HistAvgOccBetweenSet2HFM", "Average Occ Set2Between: HF-",
   MAX_LS, 0.5, (double)MAX_LS+0.5, OccBins, OccMin, OccMax );
   HistAvgOccBetweenSet2HFM->setAxisTitle( HistXTitle, 1 );
   HistAvgOccBetweenSet2HFM->setAxisTitle( HistOccYTitle, 2 );

   HistAvgOccAboveSet2HFP = dbe_->bookProfile( "HistAvgOccAboveSet2HFP", "Average Occ Set2Above: HF+",
   MAX_LS, 0.5, (double)MAX_LS+0.5, OccBins, OccMin, OccMax );
   HistAvgOccAboveSet2HFP->setAxisTitle( HistXTitle, 1 );
   HistAvgOccAboveSet2HFP->setAxisTitle( HistOccYTitle, 2 );

   HistAvgOccAboveSet2HFM = dbe_->bookProfile( "HistAvgOccAboveSet2HFM", "Average Occ Set2Above: HF-",
   MAX_LS, 0.5, (double)MAX_LS+0.5, OccBins, OccMin, OccMax );
   HistAvgOccAboveSet2HFM->setAxisTitle( HistXTitle, 1 );
   HistAvgOccAboveSet2HFM->setAxisTitle( HistOccYTitle, 2 );

   // Et Sum histories
   BXvsTimeAvgEtSumHFP  = dbe_->book2D( "BXvsTimeAvgEtSumHFP", "Average Et Sum: HF+",          
   MAX_LS, 0.5, (double)MAX_LS+0.5, NBINS, (double)XMIN, (double)XMAX);
   BXvsTimeAvgEtSumHFP->setAxisTitle( BXvsTimeXTitle, 1 );
   BXvsTimeAvgEtSumHFP->setAxisTitle( BXvsTimeYTitle, 2 );

   BXvsTimeAvgEtSumHFM  = dbe_->book2D( "BXvsTimeAvgEtSumHFM", "Average Et Sum: HF+",          
   MAX_LS, 0.5, (double)MAX_LS+0.5, NBINS, (double)XMIN, (double)XMAX);
   BXvsTimeAvgEtSumHFM->setAxisTitle( BXvsTimeXTitle, 1 );
   BXvsTimeAvgEtSumHFM->setAxisTitle( BXvsTimeYTitle, 2 );

   dbe_->setCurrentFolder(monitorName_+"/HistoryLumi");

   // Lumi Histories
   HistAvgLumiEtSum   = dbe_->bookProfile( "HistAvgLumiEtSum", "Average Instant Luminosity: Et Sum",
   MAX_LS, 0.5, (double)MAX_LS+0.5, EtSumBins, EtSumMin, EtSumMax);
   HistAvgLumiEtSum->setAxisTitle( HistXTitle, 1 );
   HistAvgLumiEtSum->setAxisTitle( HistLumiYTitle, 2 );

   HistAvgLumiOccSet1 = dbe_->bookProfile( "HistAvgLumiOccSet1", "Average Instant Luminosity: Occ Set1",
   MAX_LS, 0.5, (double)MAX_LS+0.5, OccBins, OccMin, OccMax);
   HistAvgLumiOccSet1->setAxisTitle( HistXTitle, 1 );
   HistAvgLumiOccSet1->setAxisTitle( HistLumiYTitle, 2 );

   HistAvgLumiOccSet2 = dbe_->bookProfile( "HistAvgLumiOccSet2", "Average Instant Luminosity: Occ Set2",
   MAX_LS, 0.5, (double)MAX_LS+0.5, OccBins, OccMin, OccMax);
   HistAvgLumiOccSet2->setAxisTitle( HistXTitle, 1 );
   HistAvgLumiOccSet2->setAxisTitle( HistLumiYTitle, 2 );

   HistInstantLumiEtSum   = dbe_->book1D( "HistInstantLumiEtSum", "Instant Luminosity: Et Sum",
   MAX_LS, 0.5, (double)MAX_LS+0.5);
   HistInstantLumiEtSum->setAxisTitle( HistXTitle, 1 );
   HistInstantLumiEtSum->setAxisTitle( HistLumiYTitle, 2 );

   HistInstantLumiOccSet1 = dbe_->book1D( "HistInstantLumiOccSet1", "Instant Luminosity: Occ Set1",
   MAX_LS, 0.5, (double)MAX_LS+0.5);
   HistInstantLumiOccSet1->setAxisTitle( HistXTitle, 1 );
   HistInstantLumiOccSet1->setAxisTitle( HistLumiYTitle, 2 );

   HistInstantLumiOccSet2 = dbe_->book1D( "HistInstantLumiOccSet2", "Instant Luminosity: Occ Set2",
   MAX_LS, 0.5, (double)MAX_LS+0.5);
   HistInstantLumiOccSet2->setAxisTitle( HistXTitle, 1 );
   HistInstantLumiOccSet2->setAxisTitle( HistLumiYTitle, 2 );

   HistIntegratedLumiEtSum   = dbe_->book1D( "HistIntegratedLumiEtSum", "Integrated Luminosity: Et Sum",
   MAX_LS, 0.5, (double)MAX_LS+0.5);
   HistIntegratedLumiEtSum->setAxisTitle( HistXTitle, 1 );
   HistIntegratedLumiEtSum->setAxisTitle( HistLumiYTitle, 2 );

   HistIntegratedLumiOccSet1 = dbe_->book1D( "HistIntegratedLumiOccSet1", "Integrated Luminosity: Occ Set1",
   MAX_LS, 0.5, (double)MAX_LS+0.5);
   HistIntegratedLumiOccSet1->setAxisTitle( HistXTitle, 1 );
   HistIntegratedLumiOccSet1->setAxisTitle( HistLumiYTitle, 2 );

   HistIntegratedLumiOccSet2 = dbe_->book1D( "HistIntegratedLumiOccSet2", "Integrated Luminosity: Occ Set2",
   MAX_LS, 0.5, (double)MAX_LS+0.5);
   HistIntegratedLumiOccSet2->setAxisTitle( HistXTitle, 1 );
   HistIntegratedLumiOccSet2->setAxisTitle( HistLumiYTitle, 2 );

   dbe_->setCurrentFolder(monitorName_+"/RecentHistoryLumi");

   // Lumi Recent Histories (past 128 short sections)
   RecentInstantLumiEtSum   = dbe_->book1D( "RecentInstantLumiEtSum", "Instant Luminosity: Et Sum",
   128, 0.5, (double)128+0.5);
   RecentInstantLumiEtSum->setAxisTitle( RecentHistXTitle, 1 );
   RecentInstantLumiEtSum->setAxisTitle( HistLumiYTitle, 2 );

   RecentInstantLumiOccSet1 = dbe_->book1D( "RecentInstantLumiOccSet1", "Instant Luminosity: Occ Set1",
   128, 0.5, (double)128+0.5);
   RecentInstantLumiOccSet1->setAxisTitle( RecentHistXTitle, 1 );
   RecentInstantLumiOccSet1->setAxisTitle( HistLumiYTitle, 2 );

   RecentInstantLumiOccSet2 = dbe_->book1D( "RecentInstantLumiOccSet2", "Instant Luminosity: Occ Set2",
   128, 0.5, (double)128+0.5);
   RecentInstantLumiOccSet2->setAxisTitle( RecentHistXTitle, 1 );
   RecentInstantLumiOccSet2->setAxisTitle( HistLumiYTitle, 2 );

   RecentIntegratedLumiEtSum   = dbe_->book1D( "RecentIntegratedLumiEtSum", "Integrated Luminosity: Et Sum",
   128, 0.5, (double)128+0.5);
   RecentIntegratedLumiEtSum->setAxisTitle( RecentHistXTitle, 1 );
   RecentIntegratedLumiEtSum->setAxisTitle( HistLumiYTitle, 2 );

   RecentIntegratedLumiOccSet1 = dbe_->book1D( "RecentIntegratedLumiOccSet1", "Integrated Luminosity: Occ Set1",
   128, 0.5, (double)128+0.5);
   RecentIntegratedLumiOccSet1->setAxisTitle( RecentHistXTitle, 1 );
   RecentIntegratedLumiOccSet1->setAxisTitle( HistLumiYTitle, 2 );

   RecentIntegratedLumiOccSet2 = dbe_->book1D( "RecentIntegratedLumiOccSet2", "Integrated Luminosity: Occ Set2",
   128, 0.5, (double)128+0.5);
   RecentIntegratedLumiOccSet2->setAxisTitle( RecentHistXTitle, 1 );
   RecentIntegratedLumiOccSet2->setAxisTitle( HistLumiYTitle, 2 );

   std::vector<std::string> systems = (dbe_->cd(), dbe_->getSubdirs());
   for( size_t i=0, e = systems.size(); i<e; ++i ){
      std::cout << "Systems " << systems[i] << std::endl;
   }
 
   dbe_->showDirStructure();
}


void HLXMonitor::SetupEventInfo( )
{

   using std::string;

   string currentfolder = subSystemName_ + "/" +  eventInfoFolder_;
   //cout << "currentfolder " << currentfolder << endl;

   dbe_->setCurrentFolder(currentfolder) ;

   //Event specific contents
   runId_     = dbe_->bookInt("iRun");
   lumisecId_ = dbe_->bookInt("iLumiSection");

   reportSummary_ = dbe_->bookFloat("reportSummary");
   reportSummaryMap_ = dbe_->book2D("reportSummaryMap", "reportSummaryMap", 18, 0., 18., 2, -1.5, 1.5);

   // Fill the report summary objects with default values, since these will only
   // be filled at the change of run.
   reportSummary_->Fill(1.0);

   for( unsigned int iHLX = 0; iHLX < NUM_HLX; ++iHLX ){
      unsigned int iWedge = HLXHFMap[iHLX] + 1;
      unsigned int iEta = 2;
      if( iWedge >= 19 ){ iEta = 1; iWedge -= 18; }
      reportSummaryMap_->setBinContent(iWedge,iEta,1.0);
   }   
}


// ------------ method called to for each event  ------------
void
HLXMonitor::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   while( HLXTCP.IsConnected() == false )
   {
      HLXTCP.SetIP(DistribIP1);
      if( HLXTCP.Connect() != 1 )
      {
	 std::cout << "Failed to connect to " << DistribIP1 << "." << std::endl;
	 sleep( 1 );
	 std::cout << "Trying " << DistribIP2 << std::endl;
	 HLXTCP.SetIP( DistribIP2 );
	 if( HLXTCP.Connect() == 1) break;
	 std::cout << "Failed to connect to " << DistribIP2 << "." << std::endl;
	 std::cout << " Reconnect in " << reconnTime << " seconds." <<  std::endl;
	 sleep(reconnTime);
      }
   }
   if( HLXTCP.IsConnected() == true )
   {
      std::cout << "Successfully connected." << std::endl; 
   }

   if( HLXTCP.ReceiveLumiSection( lumiSection ) == 1 )
   {
      // If this is the first time through, set the runNumber ...
      if( runNumber_ != lumiSection.hdr.runNumber ) runNumber_ = lumiSection.hdr.runNumber;
      //std::cout << "Run number is: " << runNumber_ << std::endl;
      
      // Fill the monitoring histograms 
      FillHistograms(lumiSection);
      FillHistoHFCompare(lumiSection);
      FillEventInfo(lumiSection);
      
      cout << "Run: " << lumiSection.hdr.runNumber 
	   << " Section: " << lumiSection.hdr.sectionNumber 
	   << " Orbit: " << lumiSection.hdr.startOrbit << endl;
      cout << "Et Lumi: " << lumiSection.lumiSummary.InstantETLumi << endl;
      cout << "Occ Lumi 1: " << lumiSection.lumiSummary.InstantOccLumi[0] << endl;
      cout << "Occ Lumi 2: " << lumiSection.lumiSummary.InstantOccLumi[1] << endl;
   }
   else
   {
      HLXTCP.Disconnect();
      EndRun();
   }

//    do
//    {
//       errorCode = HLXTCP.ReceiveLumiSection(lumiSection);

//       while(errorCode !=1)
//       {
// 	 HLXTCP.Disconnect();
// 	 //cout << "Connecting to TCPDistributor" << endl;
// 	 errorCode = HLXTCP.Connect();
// 	 if(errorCode != 1) 
// 	 {
//  	   if( (attemptCounter%10)==0 ){  
// 	     cout << "*** Connection Failed: " << errorCode 
// 		  << " Will attempt to reconnect in " << reconnTime << " seconds." << endl;
// 	     cout << "This message will be printed once every 10 attempts." << endl;
//  	   }
//  	   ++attemptCounter;
// 	   sleep(reconnTime);
// 	 }
//       }    
//    } while( errorCode != 1 );

}


void HLXMonitor::SaveDQMFile(){

  std::ostringstream tempStreamer;
  tempStreamer << OutputDir << "/" << OutputFilePrefix << "_" << subSystemName_
	       << "_R" << std::setfill('0') << std::setw(runNumLength) 
	       << runNumber_ << "_T00000001.root";

  std::vector<std::string> systems = (dbe_->cd(), dbe_->getSubdirs());
  char rewrite[64]; sprintf(rewrite, "\\1Run %d/\\2/Run summary", runNumber_);
  int saveReference_ = DQMStore::SaveWithoutReference;
  int saveReferenceQMin_ = dqm::qstatus::STATUS_OK;

  for( size_t i = 0, e = systems.size(); i != e; ++i )
     if (systems[i] != "Reference")
         dbe_->save( tempStreamer.str(), systems[i], "^(Reference/)?([^/]+)", rewrite,
		     (DQMStore::SaveReferenceTag)saveReference_, saveReferenceQMin_);

  //dbe_->save(tempStreamer.str());
}

// ------------ method called once each job just before starting event loop  ------------
void HLXMonitor::beginJob(const edm::EventSetup&)
{ 
   HLXTCP.SetIP(DistribIP1);
   int errorCode = HLXTCP.SetPort(listenPort);
   cout << "SetPort: " << listenPort << " Success: " << errorCode << endl;
   errorCode = HLXTCP.SetMode(AquireMode);
   cout << "AquireMode: " << AquireMode << " Success: " << errorCode << endl;

   while( HLXTCP.IsConnected() == false )
   {
      HLXTCP.SetIP(DistribIP1);
      if( HLXTCP.Connect() != 1 )
      {
	 std::cout << "Failed to connect to " << DistribIP1 << "." << std::endl;
	 sleep( 1 );
	 std::cout << "Trying " << DistribIP2 << std::endl;
	 HLXTCP.SetIP( DistribIP2 );
	 if( HLXTCP.Connect() == 1) break;
	 std::cout << "Failed to connect to " << DistribIP2 << "." << std::endl;
	 std::cout << " Reconnect in " << reconnTime << " seconds." <<  std::endl;
	 sleep(reconnTime);
      }
   }
   if( HLXTCP.IsConnected() == true )
   {
      std::cout << "Successfully connected." << std::endl; 
   }

//    do
//    {
//       //cout << "BEGINJOB: Connecting to TCPDistributor" << endl;
//       errorCode = HLXTCP.Connect();
//       //cout << "ErrorCode " << errorCode << endl;
//       if(errorCode != 1)
//       {
//  	if( (attemptCounter%10)==0 ){
// 	  cout << "BeginJob: Attempting to reconnect in " << reconnTime << " seconds." << endl;
// 	  cout << "This message will be printed once every 10 attempts." << endl;
//  	}
//  	++attemptCounter;
// 	sleep(reconnTime);
//       }
//    } while(errorCode != 1);
}

// ------------ method called once each job just after ending the event loop  ------------
void HLXMonitor::endJob() 
{
   // Fill the report summaries at end job
   // Loop over the HLX's and fill the map, 
   // also calculate the overall quality.
   HLXTCP.Disconnect();
   EndRun( SaveAtEndJob );
}

void HLXMonitor::EndRun( bool saveFile )
{
   // Run summary - Loop over the HLX's and fill the map, 
   // also calculate the overall quality.
   float overall = 0.0;
   for( unsigned int iHLX = 0; iHLX < NUM_HLX; ++iHLX ){
      unsigned int iWedge = HLXHFMap[iHLX] + 1;
      unsigned int iEta = 2;
      if( iWedge >= 19 ){ iEta = 1; iWedge -= 18; }
      float frac = (float)totalNibbles_[iWedge-1]/(float)expectedNibbles_; 
      reportSummaryMap_->setBinContent(iWedge,iEta,frac);
      overall += frac;
   }   
      
   overall /= (float)NUM_HLX;
   if( overall > 1.0 ) overall = 0.0;
   //std::cout << "Filling report summary! Main. " << overall << std::endl;
   reportSummary_->Fill(overall);
   
   // Do some things that should be done at the end of the run ...
   if( saveFile ) SaveDQMFile();  
   expectedNibbles_ = 0;
   for( unsigned int iHLX = 0; iHLX < NUM_HLX; ++iHLX ) totalNibbles_[iHLX] = 0;
   
   std::cout << "** Here in end run **" << std::endl;
   if(ResetAtNewRun) ResetAll();
   runNumber_ = 0;
}


void HLXMonitor::FillHistograms(const LUMI_SECTION & section)
{
   int lsBin = int(lumiSectionCount/64);
   int lsBinBX = int(lumiSectionCount/64);
   HistAvgLumiEtSum->Fill(lsBin, section.lumiSummary.InstantETLumi);
   HistAvgLumiOccSet1->Fill(lsBin, section.lumiSummary.InstantOccLumi[0]);
   HistAvgLumiOccSet2->Fill(lsBin, section.lumiSummary.InstantOccLumi[1]);

   int fillBin = lumiSectionCount+1;
   if( fillBin > 128 )
   {
      for( int iBin = 1; iBin<128; iBin++ )
      {
	 RecentInstantLumiEtSum->setBinContent(iBin,RecentInstantLumiEtSum->getBinContent(iBin));
	 RecentInstantLumiOccSet1->setBinContent(iBin,RecentInstantLumiOccSet1->getBinContent(iBin));
	 RecentInstantLumiOccSet2->setBinContent(iBin,RecentInstantLumiOccSet2->getBinContent(iBin));
	 RecentIntegratedLumiEtSum->setBinContent(iBin,RecentIntegratedLumiEtSum->getBinContent(iBin));
	 RecentIntegratedLumiOccSet1->setBinContent(iBin,RecentIntegratedLumiOccSet1->getBinContent(iBin));
	 RecentIntegratedLumiOccSet2->setBinContent(iBin,RecentIntegratedLumiOccSet2->getBinContent(iBin));
      }
      fillBin = 128;
   }

   RecentInstantLumiEtSum->setBinContent(fillBin,sectionInstantSumEt);
   RecentInstantLumiEtSum->setBinError(fillBin,sqrt(sectionInstantErrSumEt));
   RecentInstantLumiOccSet1->setBinContent(fillBin,sectionInstantSumOcc1);
   RecentInstantLumiOccSet1->setBinError(fillBin,sqrt(sectionInstantErrSumOcc1));
   RecentInstantLumiOccSet2->setBinContent(fillBin,sectionInstantSumOcc2);
   RecentInstantLumiOccSet2->setBinError(fillBin,sqrt(sectionInstantErrSumOcc2));
      
   double recentOldBinContent = RecentIntegratedLumiEtSum->getBinContent(fillBin-1);
   double recentNewBinContent = recentOldBinContent + sectionInstantSumEt; 
   RecentIntegratedLumiEtSum->setBinContent(fillBin,recentNewBinContent);
   recentOldBinContent = RecentIntegratedLumiOccSet1->getBinContent(fillBin-1);
   recentNewBinContent = recentOldBinContent + sectionInstantSumOcc1; 
   RecentIntegratedLumiOccSet1->setBinContent(fillBin,recentNewBinContent);
   recentOldBinContent = RecentIntegratedLumiOccSet2->getBinContent(fillBin-1-1);
   recentNewBinContent = recentOldBinContent + sectionInstantSumOcc2; 
   RecentIntegratedLumiOccSet2->setBinContent(fillBin,recentNewBinContent);

   double recentOldBinError = RecentIntegratedLumiEtSum->getBinError(fillBin-1);
   double recentNewBinError = sqrt(recentOldBinError*recentOldBinError + sectionInstantErrSumEt); 
   RecentIntegratedLumiEtSum->setBinError(fillBin,recentNewBinError);
   recentOldBinError = RecentIntegratedLumiOccSet1->getBinError(fillBin-1);
   recentNewBinError = sqrt(recentOldBinError*recentOldBinError + sectionInstantErrSumOcc1); 
   RecentIntegratedLumiOccSet1->setBinError(fillBin,recentNewBinError);
   recentOldBinError = RecentIntegratedLumiOccSet2->getBinError(fillBin-1);
   recentNewBinError = sqrt(recentOldBinError*recentOldBinError + sectionInstantErrSumOcc2); 
   RecentIntegratedLumiOccSet2->setBinError(fillBin,recentNewBinError);

   if( lsBinOld != lsBin )
   {
      HistInstantLumiEtSum->setBinContent(lsBinOld,sectionInstantSumEt);
      HistInstantLumiEtSum->setBinError(lsBinOld,sqrt(sectionInstantErrSumEt));
      HistInstantLumiOccSet1->setBinContent(lsBinOld,sectionInstantSumOcc1);
      HistInstantLumiOccSet1->setBinError(lsBinOld,sqrt(sectionInstantErrSumOcc1));
      HistInstantLumiOccSet2->setBinContent(lsBinOld,sectionInstantSumOcc2);
      HistInstantLumiOccSet2->setBinError(lsBinOld,sqrt(sectionInstantErrSumOcc2));
      
      double histOldBinContent = HistIntegratedLumiEtSum->getBinContent(lsBinOld-1);
      double histNewBinContent = histOldBinContent + sectionInstantSumEt; 
      HistIntegratedLumiEtSum->setBinContent(lsBinOld,histNewBinContent);
      histOldBinContent = HistIntegratedLumiOccSet1->getBinContent(lsBinOld-1);
      histNewBinContent = histOldBinContent + sectionInstantSumOcc1; 
      HistIntegratedLumiOccSet1->setBinContent(lsBinOld,histNewBinContent);
      histOldBinContent = HistIntegratedLumiOccSet2->getBinContent(lsBinOld-1);
      histNewBinContent = histOldBinContent + sectionInstantSumOcc2; 
      HistIntegratedLumiOccSet2->setBinContent(lsBinOld,histNewBinContent);

      double histOldBinError = HistIntegratedLumiEtSum->getBinError(lsBinOld-1);
      double histNewBinError = sqrt(histOldBinError*histOldBinError + sectionInstantErrSumEt); 
      HistIntegratedLumiEtSum->setBinError(lsBinOld,histNewBinError);
      histOldBinError = HistIntegratedLumiOccSet1->getBinError(lsBinOld-1);
      histNewBinError = sqrt(histOldBinError*histOldBinError + sectionInstantErrSumOcc1); 
      HistIntegratedLumiOccSet1->setBinError(lsBinOld,histNewBinError);
      histOldBinError = HistIntegratedLumiOccSet2->getBinError(lsBinOld-1);
      histNewBinError = sqrt(histOldBinError*histOldBinError + sectionInstantErrSumOcc2); 
      HistIntegratedLumiOccSet2->setBinError(lsBinOld,histNewBinError);

      sectionInstantSumEt = 0;
      sectionInstantErrSumEt = 0;
      sectionInstantSumOcc1 = 0;
      sectionInstantErrSumOcc1 = 0;
      sectionInstantSumOcc2 = 0;
      sectionInstantErrSumOcc2 = 0;
      sectionInstantNorm = 0;
      lsBinOld = lsBin;
   }

   sectionInstantSumEt += section.lumiSummary.InstantETLumi;
   sectionInstantErrSumEt += section.lumiSummary.InstantETLumiErr*section.lumiSummary.InstantETLumiErr;
   sectionInstantSumOcc1 += section.lumiSummary.InstantOccLumi[0];
   sectionInstantErrSumOcc1 += section.lumiSummary.InstantOccLumiErr[0]*section.lumiSummary.InstantOccLumiErr[0];
   sectionInstantSumOcc2 += section.lumiSummary.InstantOccLumi[1];
   sectionInstantErrSumOcc2 += section.lumiSummary.InstantOccLumiErr[1]*section.lumiSummary.InstantOccLumiErr[1];
   ++sectionInstantNorm;
   
   dbe_->softReset(LumiInstantEtSum);
   dbe_->softReset(LumiInstantOccSet1);
   dbe_->softReset(LumiInstantOccSet2);

   for( int iHLX = 0; iHLX < (int)NUM_HLX; ++iHLX )
   {
      unsigned int utotal1= 0;
      unsigned int utotal2 = 0;
      unsigned int iWedge = HLXHFMap[iHLX];
      if(section.occupancy[iHLX].hdr.numNibbles != 0)
      {
	 for( unsigned int iBX = 0; iBX < NUM_BUNCHES; ++iBX )  // Don't include the last one hundred BX in the average.
	 {
	    // Normalize to number of towers
	    unsigned int norm[2] = {0,0};
	    norm[0] += section.occupancy[iHLX].data[set1BelowIndex   ][iBX];
	    norm[0] += section.occupancy[iHLX].data[set1BetweenIndex ][iBX];
	    norm[0] += section.occupancy[iHLX].data[set1AboveIndex   ][iBX];
	    if( norm[0] == 0 ) continue;
	    norm[1] += section.occupancy[iHLX].data[set2BelowIndex   ][iBX];
	    norm[1] += section.occupancy[iHLX].data[set2BetweenIndex ][iBX];
	    norm[1] += section.occupancy[iHLX].data[set2AboveIndex   ][iBX];
	    if( norm[1] == 0 ) continue;

	    double normEt = section.etSum[iHLX].data[iBX]/(double)(norm[0]+norm[1]);
	    double normOccSet1Below   = (double)section.occupancy[iHLX].data[set1BelowIndex][iBX]/(double)norm[0];
	    double normOccSet1Between = (double)section.occupancy[iHLX].data[set1BetweenIndex][iBX]/(double)norm[0];
	    double normOccSet1Above   = (double)section.occupancy[iHLX].data[set1AboveIndex][iBX]/(double)norm[0];
	    double normOccSet2Below   = (double)section.occupancy[iHLX].data[set2BelowIndex][iBX]/(double)norm[1];
	    double normOccSet2Between = (double)section.occupancy[iHLX].data[set2BetweenIndex][iBX]/(double)norm[1];
	    double normOccSet2Above   = (double)section.occupancy[iHLX].data[set2AboveIndex][iBX]/(double)norm[1];

	    // Averages & check sum
	    if( iBX < NUM_BUNCHES-100 )
	    {
	       AvgEtSum->Fill( iWedge,normEt);
	
	       AvgOccBelowSet1->  Fill( iWedge, normOccSet1Below   );
	       AvgOccBetweenSet1->Fill( iWedge, normOccSet1Between );
	       AvgOccAboveSet1->  Fill( iWedge, normOccSet1Above   );
	   
	       AvgOccBelowSet2->  Fill( iWedge, normOccSet2Below    );
	       AvgOccBetweenSet2->Fill( iWedge, normOccSet2Between  );
	       AvgOccAboveSet2->  Fill( iWedge, normOccSet2Above    );

	       if( iWedge < 18 )
	       {
		  HistAvgEtSumHFP->Fill( lsBin,normEt);
		  HistAvgOccBelowSet1HFP->Fill( lsBin,   normOccSet1Below    );
		  HistAvgOccBetweenSet1HFP->Fill( lsBin, normOccSet1Between  );
		  HistAvgOccAboveSet1HFP->Fill( lsBin,   normOccSet1Above    );
		  HistAvgOccBelowSet2HFP->Fill( lsBin,   normOccSet2Below    );
		  HistAvgOccBetweenSet2HFP->Fill( lsBin, normOccSet2Between  );
		  HistAvgOccAboveSet2HFP->Fill( lsBin,   normOccSet2Above    );

		  if( iBX >= (XMIN-1) && iBX <= (XMAX-1) ) BXvsTimeAvgEtSumHFP->Fill(lsBinBX,iBX,normEt/(64.0*18.0*12.0));
	       }
	       else
	       {
		  HistAvgEtSumHFM->Fill( lsBin,normEt);
		  HistAvgOccBelowSet1HFM->Fill( lsBin,   normOccSet1Below    );
		  HistAvgOccBetweenSet1HFM->Fill( lsBin, normOccSet1Between  );
		  HistAvgOccAboveSet1HFM->Fill( lsBin,   normOccSet1Above    );
		  HistAvgOccBelowSet2HFM->Fill( lsBin,   normOccSet2Below    );
		  HistAvgOccBetweenSet2HFM->Fill( lsBin, normOccSet2Between  );
		  HistAvgOccAboveSet2HFM->Fill( lsBin,   normOccSet2Above    );

		  if( iBX >= (XMIN-1) && iBX <= (XMAX-1) ) BXvsTimeAvgEtSumHFM->Fill(lsBinBX,iBX,normEt/(64.0*18.0*12.0));
	       }

	       utotal1 += section.occupancy[iHLX].data[set1BelowIndex  ][iBX];
	       utotal1 += section.occupancy[iHLX].data[set1BetweenIndex][iBX];
	       utotal1 += section.occupancy[iHLX].data[set1AboveIndex  ][iBX];

	       utotal2 += section.occupancy[iHLX].data[set2BelowIndex  ][iBX];
	       utotal2 += section.occupancy[iHLX].data[set2BetweenIndex][iBX];
	       utotal2 += section.occupancy[iHLX].data[set2AboveIndex  ][iBX];

	    }

	    if(Style.compare("BX") == 0)
	    {
	       Set1Below[iWedge]->  Fill(iBX, normOccSet1Below   );
	       Set1Between[iWedge]->Fill(iBX, normOccSet1Between );
	       Set1Above[iWedge]->  Fill(iBX, normOccSet1Above   );
	       Set2Below[iWedge]->  Fill(iBX, normOccSet2Below   );
	       Set2Between[iWedge]->Fill(iBX, normOccSet2Between );
	       Set2Above[iWedge]->  Fill(iBX, normOccSet2Above   );
	       ETSum[iWedge]->      Fill(iBX, normEt);
	    }
	    else if(Style.compare("Dist")==0)
	    {
	       Set1Below[iWedge]->  Fill( normOccSet1Below    );
	       Set1Between[iWedge]->Fill( normOccSet1Between  );
	       Set1Above[iWedge]->  Fill( normOccSet1Above    );
	       Set2Below[iWedge]->  Fill( normOccSet2Below    );
	       Set2Between[iWedge]->Fill( normOccSet2Between  );
	       Set2Above[iWedge]->  Fill( normOccSet2Above    );
	       ETSum[iWedge]->      Fill( normEt );
	    }


	    LumiAvgEtSum->Fill(iBX, section.lumiDetail.ETLumi[iBX]);
	    LumiAvgOccSet1->Fill(iBX, section.lumiDetail.OccLumi[0][iBX]);
	    LumiAvgOccSet2->Fill(iBX, section.lumiDetail.OccLumi[1][iBX]);

	    int iBin = iBX - (int)XMIN + 1;
	    if( iBin <= int(XMAX-XMIN) && iBin >= 1 )
	    {
	      LumiInstantEtSum->setBinContent(iBin, section.lumiDetail.ETLumi[iBX]);
	      LumiInstantOccSet1->setBinContent(iBin, section.lumiDetail.OccLumi[0][iBX]);
	      LumiInstantOccSet2->setBinContent(iBin, section.lumiDetail.OccLumi[1][iBX]);
	      LumiInstantEtSum->setBinError(iBin, section.lumiDetail.ETLumiErr[iBX]);
	      LumiInstantOccSet1->setBinError(iBin, section.lumiDetail.OccLumiErr[0][iBX]);
	      LumiInstantOccSet2->setBinError(iBin, section.lumiDetail.OccLumiErr[1][iBX]);

	      double oldBinContent = LumiIntegratedEtSum->getBinContent(iBin);
	      double newBinContent = oldBinContent + section.lumiDetail.ETLumi[iBX];
	      LumiIntegratedEtSum->setBinContent(iBin, newBinContent);
	      oldBinContent = LumiIntegratedOccSet1->getBinContent(iBin);
	      newBinContent = oldBinContent + section.lumiDetail.OccLumi[0][iBX];
	      LumiIntegratedOccSet1->setBinContent(iBin, newBinContent);
	      oldBinContent = LumiIntegratedOccSet2->getBinContent(iBin);
	      newBinContent = oldBinContent + section.lumiDetail.OccLumi[1][iBX];
	      LumiIntegratedOccSet2->setBinContent(iBin, newBinContent);

	      double oldBinError = LumiIntegratedEtSum->getBinError(iBin);
	      double newBinError = sqrt(oldBinError*oldBinError + section.lumiDetail.ETLumiErr[iBX]*section.lumiDetail.ETLumiErr[iBX]);
	      LumiIntegratedEtSum->setBinError(iBin, newBinError);
	      oldBinError = LumiIntegratedOccSet1->getBinError(iBin);
	      newBinError = sqrt(oldBinError*oldBinError + section.lumiDetail.OccLumiErr[0][iBX]*section.lumiDetail.OccLumiErr[0][iBX]);
	      LumiIntegratedOccSet1->setBinError(iBin, newBinError);
	      oldBinError = LumiIntegratedOccSet2->getBinError(iBin);
	      newBinError = sqrt(oldBinError*oldBinError + section.lumiDetail.OccLumiErr[1][iBX]*section.lumiDetail.OccLumiErr[1][iBX]);
	      LumiIntegratedOccSet2->setBinError(iBin, newBinError);
	    }

	 }

	 // Get the number of towers per wedge per BX (assuming non-zero numbers)
	 double total1 = 0;
	 double total2 = 0;
	 if( (NUM_BUNCHES-100)>0 )
	 {
	    total1 = (double)utotal1/(double)(NUM_BUNCHES-100);
	    total2 = (double)utotal2/(double)(NUM_BUNCHES-100);
	 }
	 if( section.hdr.numOrbits > 0 ) 
	 {
	    total1 = total1/(double)section.hdr.numOrbits;
	    total2 = total2/(double)section.hdr.numOrbits;
	 }

	 SumAllOccSet1->  Fill( iWedge, total1 );
	 SumAllOccSet2->  Fill( iWedge, total2 );

      }
   }

   // Add one to the section count (usually short sections)
   ++lumiSectionCount;

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
   // New run .. set the run number and fill run summaries ...
   std::cout << "Run number " << runNumber_ << " Section hdr run number " 
	     << section.hdr.runNumber << std::endl;

   runId_->Fill( section.hdr.runNumber );
   lumisecId_->Fill( (int)(section.hdr.sectionNumber/64) + 1 );

   // Update the total nibbles & the expected number
   expectedNibbles_ += 4;
   for( unsigned int iHLX = 0; iHLX < NUM_HLX; ++iHLX ){
      unsigned int iWedge = HLXHFMap[iHLX] + 1;
      totalNibbles_[iWedge-1] += section.occupancy[iHLX].hdr.numNibbles; 
   }   

}

void HLXMonitor::ResetAll()
{
   for( unsigned int iHLX = 0; iHLX < NUM_HLX; ++iHLX )
   {
      dbe_->softReset( Set1Below[iHLX] );
      dbe_->softReset( Set1Between[iHLX] );
      dbe_->softReset( Set1Above[iHLX] );
      dbe_->softReset( Set1Below[iHLX] );
      dbe_->softReset( Set1Between[iHLX] );
      dbe_->softReset( Set1Above[iHLX] );
	   
      dbe_->softReset( ETSum[iHLX] );
	   
   }
	 
   dbe_->softReset(HFCompareEtSum);
   dbe_->softReset(HFCompareOccBelowSet1);
   dbe_->softReset(HFCompareOccBetweenSet1);
   dbe_->softReset(HFCompareOccAboveSet1);
   dbe_->softReset(HFCompareOccBelowSet2);
   dbe_->softReset(HFCompareOccBetweenSet2);
   dbe_->softReset(HFCompareOccAboveSet2);

   dbe_->softReset(AvgEtSum);
   dbe_->softReset(AvgOccBelowSet1);
   dbe_->softReset(AvgOccBetweenSet1);
   dbe_->softReset(AvgOccAboveSet1);
   dbe_->softReset(AvgOccBelowSet2);
   dbe_->softReset(AvgOccBetweenSet2);
   dbe_->softReset(AvgOccAboveSet2);

   // Luminosity Monitoring
   dbe_->softReset(LumiAvgEtSum);
   dbe_->softReset(LumiAvgOccSet1);
   dbe_->softReset(LumiAvgOccSet2);
   dbe_->softReset(LumiInstantEtSum);
   dbe_->softReset(LumiInstantOccSet1);
   dbe_->softReset(LumiInstantOccSet2);
   dbe_->softReset(LumiIntegratedEtSum);
   dbe_->softReset(LumiIntegratedOccSet1);
   dbe_->softReset(LumiIntegratedOccSet2);

   // Sanity Check for Occupancy
   dbe_->softReset(SumAllOccSet1);
   dbe_->softReset(SumAllOccSet2);

   // History 
   lumiSectionCount = 0;
   dbe_->softReset(HistAvgEtSumHFP);
   dbe_->softReset(HistAvgEtSumHFM);

   dbe_->softReset(HistAvgOccBelowSet1HFP);
   dbe_->softReset(HistAvgOccBelowSet1HFM);
   dbe_->softReset(HistAvgOccBetweenSet1HFP);
   dbe_->softReset(HistAvgOccBetweenSet1HFM);
   dbe_->softReset(HistAvgOccAboveSet1HFP);
   dbe_->softReset(HistAvgOccAboveSet1HFM);

   dbe_->softReset(HistAvgOccBelowSet2HFP);
   dbe_->softReset(HistAvgOccBelowSet2HFM);
   dbe_->softReset(HistAvgOccBetweenSet2HFP);
   dbe_->softReset(HistAvgOccBetweenSet2HFM);
   dbe_->softReset(HistAvgOccAboveSet2HFP);
   dbe_->softReset(HistAvgOccAboveSet2HFM);

   dbe_->softReset(HistAvgLumiEtSum);
   dbe_->softReset(HistAvgLumiOccSet1);
   dbe_->softReset(HistAvgLumiOccSet2);
   dbe_->softReset(HistInstantLumiEtSum);
   dbe_->softReset(HistInstantLumiOccSet1);
   dbe_->softReset(HistInstantLumiOccSet2);
   dbe_->softReset(HistIntegratedLumiEtSum);
   dbe_->softReset(HistIntegratedLumiOccSet1);
   dbe_->softReset(HistIntegratedLumiOccSet2);

   dbe_->softReset(RecentInstantLumiEtSum);
   dbe_->softReset(RecentInstantLumiOccSet1);
   dbe_->softReset(RecentInstantLumiOccSet2);
   dbe_->softReset(RecentIntegratedLumiEtSum);
   dbe_->softReset(RecentIntegratedLumiOccSet1);
   dbe_->softReset(RecentIntegratedLumiOccSet2);

   dbe_->softReset(BXvsTimeAvgEtSumHFP);
   dbe_->softReset(BXvsTimeAvgEtSumHFM);
}



//define this as a plug-in
DEFINE_FWK_MODULE(HLXMonitor);
