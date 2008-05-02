/*
    Author:  Adam Hunt
    email:   ahunt@princeton.edu
*/
#include "DQM/HLXMonitor/interface/HLXMonitor.h"

using std::cout;
using std::endl;

HLXMonitor::HLXMonitor(const edm::ParameterSet& iConfig)
{

   NUM_HLX       = iConfig.getUntrackedParameter<unsigned int>("numHlx", 36);
   NUM_BUNCHES   = iConfig.getUntrackedParameter<unsigned int>("numBunches",3564);
   listenPort    = iConfig.getUntrackedParameter<unsigned int>("SourcePort", 51001);
   OutputFile    = iConfig.getUntrackedParameter<std::string>("outputFile","lumi");
   OutputDir     = iConfig.getUntrackedParameter<std::string>("outputDir","data");
   SavePeriod    = iConfig.getUntrackedParameter<unsigned int>("SavePeriod",10);
   NBINS         = iConfig.getUntrackedParameter<unsigned int>("NBINS",3564);
   XMIN          = iConfig.getUntrackedParameter<double>("XMIN",0);
   XMAX          = iConfig.getUntrackedParameter<double>("XMAX",3564);
   Style         = iConfig.getUntrackedParameter<std::string>("Style","BX");
   AquireMode    = iConfig.getUntrackedParameter<unsigned int>("AquireMode",0);
   Accumulate    = iConfig.getUntrackedParameter<bool>("Accumulate",true); // all
   TriggerBX     = iConfig.getUntrackedParameter<unsigned int>("TriggerBX",50);
   reconnTime    = iConfig.getUntrackedParameter<unsigned int>("ReconnectionTime",5);
   DistribIP     = iConfig.getUntrackedParameter<std::string>("HLXIP","localhost");

   // HLX Config info
   set1BelowIndex = 0;
   set1BetweenIndex = 1;
   set1AboveIndex = 2;
   set2BelowIndex = 3;
   set2BetweenIndex = 4;
   set2AboveIndex = 5;

   runNumLength = 9;
   secNumLength = 6;

   if(NUM_HLX > 36)       NUM_HLX = 36;

   if(NUM_BUNCHES > 3564) NUM_BUNCHES = 3564;

   if(XMAX <= XMIN)
   {
      XMIN = 0;
      if(XMAX <= 0) XMAX = 3564;
   }

   if((Style.compare("History")==0) || (NBINS == 0))
   {
      NBINS = (unsigned int)(XMAX-XMIN);
   }

   // get hold of back-end interface
   // dbe_ = edm::Service<DaqMonitorBEInterface>().operator->();

   dbe_ = edm::Service<DQMStore>().operator->();

   if ( dbe_ ) {
     dbe_->setVerbose(1);
   }

   monitorName_ = iConfig.getUntrackedParameter<std::string>("monitorName","HLX");
   cout << "Monitor name = " << monitorName_ << endl;
   //   if( monitorName_ != "" ) monitorName_ = monitorName_+"/" ;
   prescaleEvt_ = iConfig.getUntrackedParameter<int>("prescaleEvt", -1);
   cout << "===>DQM event prescale = " << prescaleEvt_ << " events "<< endl;

   //We put this here for the moment since there is no better place 
   //  edm::Service<MonitorDaemon> daemon;
   // daemon.operator->();

   // Setup the monitoring histograms
   SetupHists();
  
}

HLXMonitor::~HLXMonitor()
{
  HT.Disconnect();
}

// ------------ Setup the monitoring elements ---------------
void
HLXMonitor::SetupHists()
{

  dbe_->setCurrentFolder(monitorName_+"/HFPlus");

   for( unsigned int i=0; i < 18 && i < NUM_HLX; ++i )
   {  
      std::ostringstream tempStreamer;
      tempStreamer << std::dec << (i+1);

      dbe_->setCurrentFolder(monitorName_+"/HFPlus/Wedge"+tempStreamer.str());

      Set1Below[i]  = dbe_->book1D("Set1_Below",   "HF+ Wedge "+tempStreamer.str()+" Set1 Below Threshold",  NBINS, XMIN, XMAX);
      Set1Between[i]= dbe_->book1D("Set1_Between", "HF+ Wedge "+tempStreamer.str()+" Set1 Between Threshold",NBINS, XMIN, XMAX);
      Set1Above[i]  = dbe_->book1D("Set1_Above",   "HF+ Wedge "+tempStreamer.str()+" Set1 Above Threshold",  NBINS, XMIN, XMAX);
      Set2Below[i]  = dbe_->book1D("Set2_Below",   "HF+ Wedge "+tempStreamer.str()+" Set2 Below Threshold",  NBINS, XMIN, XMAX);
      Set2Between[i]= dbe_->book1D("Set2_Between", "HF+ Wedge "+tempStreamer.str()+" Set2 Between Threshold",NBINS, XMIN, XMAX);
      Set2Above[i]  = dbe_->book1D("Set2_Above",   "HF+ Wedge "+tempStreamer.str()+" Set2 Above Threshold",  NBINS, XMIN, XMAX);    
      ETSum[i]      = dbe_->book1D("ETSum",        "HF+ Wedge "+tempStreamer.str()+" Et Sum",                NBINS, XMIN, XMAX);    

      dbe_->tagContents(monitorName_+"/HFPlus/Wedge"+tempStreamer.str(), i+1);
   }

   if(NUM_HLX > 17)
   {
      dbe_->setCurrentFolder(monitorName_+"/HFMinus");
    
      for( unsigned int i=18; i < NUM_HLX; ++i )
      {
	 std::ostringstream tempStreamer;
	 tempStreamer << std::dec << (i+1);

	 dbe_->setCurrentFolder(monitorName_+"/HFMinus/Wedge"+tempStreamer.str());
	 Set1Below[i]   = dbe_->book1D("Set1_Below",   "HF- Wedge "+tempStreamer.str()+" Set1 Below Threshold",  NBINS, XMIN, XMAX);
	 Set1Between[i] = dbe_->book1D("Set1_Between", "HF- Wedge "+tempStreamer.str()+" Set1 Between Threshold",NBINS, XMIN, XMAX);
	 Set1Above[i]   = dbe_->book1D("Set1_Above",   "HF- Wedge "+tempStreamer.str()+" Set1 Above Threshold",  NBINS, XMIN, XMAX); 
	 Set2Below[i]   = dbe_->book1D("Set2_Below",   "HF- Wedge "+tempStreamer.str()+" Set2 Below Threshold",  NBINS, XMIN, XMAX); 
	 Set2Between[i] = dbe_->book1D("Set2_Between", "HF- Wedge "+tempStreamer.str()+" Set2 Between Threshold",NBINS, XMIN, XMAX); 
	 Set2Above[i]   = dbe_->book1D("Set2_Above",   "HF- Wedge "+tempStreamer.str()+" Set2 Above Threshold",  NBINS, XMIN, XMAX); 
	 ETSum[i]       = dbe_->book1D("ETSum",        "HF- Wedge "+tempStreamer.str()+" Et Sum",                NBINS, XMIN, XMAX); 

	 dbe_->tagContents(monitorName_+"/HFMinus/Wedge"+tempStreamer.str(), i+1);
      }
   }
  
   dbe_->setCurrentFolder(monitorName_+"/HFCompare");

   HFCompareEtSum = dbe_->book1D("HFCompareEtSum","Et Sum - cyclic trigger ",36,0,36);
   HFCompareEtSum->setAxisTitle("HLX ID",1);
   HFCompareEtSum->setAxisTitle("ET Sum per active Tower",2);
 
   HFCompareOccBelowSet1 = dbe_->book1D("HFCompareOccBelowSet1","Occupancy Below Threshold - Set 1", 36, 0, 36);
   HFCompareOccBelowSet1->setAxisTitle("HLX ID",1);
   HFCompareOccBelowSet1->setAxisTitle("Occ per active Tower",2);

   HFCompareOccBetweenSet1 = dbe_->book1D("HFCompareOccBetweenSet1","Occupancy Between Threshold - Set 1", 36, 0, 36);
   HFCompareOccBetweenSet1->setAxisTitle("HLX ID",1);
   HFCompareOccBetweenSet1->setAxisTitle("Occ per active Tower",2);

   HFCompareOccAboveSet1 = dbe_->book1D("HFCompareOccAboveSet1","Occupancy Above Threshold - Set 1", 36, 0, 36);
   HFCompareOccAboveSet1->setAxisTitle("HLX ID",1);
   HFCompareOccAboveSet1->setAxisTitle("Occ per active Tower",2);

   HFCompareOccBelowSet2 = dbe_->book1D("HFCompareOccBelowSet2","Occupancy Below Threshold - Set 2", 36, 0, 36);
   HFCompareOccBelowSet2->setAxisTitle("HLX ID",1);
   HFCompareOccBelowSet2->setAxisTitle("Occ per active Tower",2);

   HFCompareOccBetweenSet2 = dbe_->book1D("HFCompareOccBetweenSet2","Occupancy Between Threshold - Set 2", 36, 0, 36);
   HFCompareOccBetweenSet2->setAxisTitle("HLX ID",1);
   HFCompareOccBetweenSet2->setAxisTitle("Occ per active Tower",2);

   HFCompareOccAboveSet2 = dbe_->book1D("HFCompareOccAboveSet2","Occupancy Above Threshold - Set 2", 36, 0, 36);
   HFCompareOccAboveSet2->setAxisTitle("HLX ID",1);
   HFCompareOccAboveSet2->setAxisTitle("Occ per active Tower",2);

   dbe_->setCurrentFolder(monitorName_+"/Average");

   AvgEtSum = dbe_->bookProfile("AvgEtSum","Average EtSum",36,0,36, 10000,0,10000,"");
   AvgEtSum->setAxisTitle("HLX ID",1);
   AvgEtSum->setAxisTitle("Avg EtSum",2);

   AvgOccBelowSet1 = dbe_->bookProfile("AvgOccBelowSet1","Average OccBelowSet1",36,0,36,10000,0,10000,"");
   AvgOccBelowSet1->setAxisTitle("HLX ID",1);
   AvgOccBelowSet1->setAxisTitle("Avg OccBelowSet1",2);

   AvgOccBetweenSet1 = dbe_->bookProfile("AvgOccBetweenSet1","Average OccBetweenSet1",36,0,36,10000,0,10000,"");
   AvgOccBetweenSet1->setAxisTitle("HLX ID",1);
   AvgOccBetweenSet1->setAxisTitle("Avg OccBetweenSet1",2);

   AvgOccAboveSet1 = dbe_->bookProfile("AvgOccAboveSet1","Average OccAboveSet1",36,0,36,10000,0,10000,"");
   AvgOccAboveSet1->setAxisTitle("HLX ID",1);
   AvgOccAboveSet1->setAxisTitle("Avg OccAboveSet1",2);

   AvgOccBelowSet2 = dbe_->bookProfile("AvgOccBelowSet2","Average OccBelowSet2",36,0,36,10000,0,10000,"");
   AvgOccBelowSet2->setAxisTitle("HLX ID",1);
   AvgOccBelowSet2->setAxisTitle("Avg OccBelowSet2",2);

   AvgOccBetweenSet2 = dbe_->bookProfile("AvgOccBetweenSet2","Average OccBetweenSet2",36,0,36,10000,0,10000,"");
   AvgOccBetweenSet2->setAxisTitle("HLX ID",1);
   AvgOccBetweenSet2->setAxisTitle("Avg OccBetweenSet2",2);

   AvgOccAboveSet2 = dbe_->bookProfile("AvgOccAboveSet2","Average OccAboveSet2",36,0,36,10000,0,10000,"");
   AvgOccAboveSet2->setAxisTitle("HLX ID",1);
   AvgOccAboveSet2->setAxisTitle("Avg OccAboveSet2",2);

   for( unsigned int i=0; i < NUM_HLX; ++i )
   {
      Set1Below[i]->setResetMe(!Accumulate);
      Set1Between[i]->setResetMe(!Accumulate);
      Set1Above[i]->setResetMe(!Accumulate);
      Set2Below[i]->setResetMe(!Accumulate);
      Set2Between[i]->setResetMe(!Accumulate);
      Set2Above[i]->setResetMe(!Accumulate);
      ETSum[i]->setResetMe(!Accumulate);    
   }
  
   if(Style.compare("BX") == 0)
   {
      OccXAxisTitle = "Bunch Crossing";
      OccYAxisTitle = "Occupancy";
      EtXAxisTitle  = "Bunch Crossing";
      EtYAxisTitle  = "Et Sum";
   }
   else if(Style.compare("Distribution")==0)
   {
      OccXAxisTitle = "Occupancy";
      OccYAxisTitle = "Count";
      EtXAxisTitle  = "Et Sum";
      EtYAxisTitle  = "Count";
   }
//    else if(Style.compare("History")==0)
//    {
//       OccXAxisTitle = "Lumi Section";
//       OccYAxisTitle = "Avg Occupancy";
//       EtXAxisTitle  = "Lumi Section";
//       EtYAxisTitle  = "Avg Et Sum Occupancy";
    
//       Set1Below[i]->setResetMe(false);
//       Set1Between[i]->setResetMe(false);
//       Set1Above[i]->setResetMe(false);
//       Set2Below[i]->setResetMe(false);
//       Set2Between[i]->setResetMe(false);
//       Set2Above[i]->setResetMe(false);
//       ETSum[i]->setResetMe(false);
    
//    }
  
   for( unsigned int i=0; i < NUM_HLX; ++i )
   {
      Set1Below[i]->setAxisTitle(OccXAxisTitle,1);
      Set1Below[i]->setAxisTitle(OccYAxisTitle, 2);
      Set1Between[i]->setAxisTitle(OccXAxisTitle,1);
      Set1Between[i]->setAxisTitle(OccYAxisTitle, 2);
      Set1Above[i]->setAxisTitle(OccXAxisTitle,1);
      Set1Above[i]->setAxisTitle(OccYAxisTitle, 2);	
      Set2Below[i]->setAxisTitle(OccXAxisTitle,1);
      Set2Below[i]->setAxisTitle(OccYAxisTitle, 2);
      Set2Between[i]->setAxisTitle(OccXAxisTitle,1);
      Set2Between[i]->setAxisTitle(OccYAxisTitle, 2);
      Set2Above[i]->setAxisTitle(OccXAxisTitle,1);
      Set2Above[i]->setAxisTitle(OccYAxisTitle, 2);	
      ETSum[i]->setAxisTitle(EtXAxisTitle,1);
      ETSum[i]->setAxisTitle(EtYAxisTitle, 2);	  
   }

   dbe_->showDirStructure();
}


// ------------ method called to for each event  ------------
void
HLXMonitor::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   int errorCode = 0;
   do
   {
      errorCode = HT.ReceiveLumiSection(lumiSection);
      cout << "ReceiveLumiSection: " << errorCode << endl;

      while(errorCode !=1)
      {
	 HT.Disconnect();
	 cout << "Connecting to TCPDistributor" << endl;
	 errorCode = HT.Connect();
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
//    else if(Style.compare("History")==0)
//    {
//       // FillHistoHistory(lumiSection);
//    }

   FillHistoHFCompare(lumiSection);

   FillHistoAvg(lumiSection);

   cout << "Run: " << lumiSection.hdr.runNumber 
	<< " Section: " << lumiSection.hdr.sectionNumber 
	<< " Orbit: " << lumiSection.hdr.startOrbit << endl;
   cout << "Et Lumi: " << lumiSection.lumiSummary.InstantETLumi << endl;
   cout << "Occ Lumi 1: " << lumiSection.lumiSummary.InstantOccLumi[0] << endl;
   cout << "Occ Lumi 2: " << lumiSection.lumiSummary.InstantOccLumi[1] << endl;
   cout << "Noise[0]: " << lumiSection.lumiSummary.lumiNoise[0] << endl;
   cout << "Noise[1]: " << lumiSection.lumiSummary.lumiNoise[1] << endl;

   /*

   if( lumiSection.hdr.sectionNumber % SavePeriod == 1 )
   {
      std::ostringstream tempStreamer;
      std::ostringstream dqmToHtmlCommand;
      tempStreamer << OutputDir << "/" << OutputFile 
		   << "_" << std::setfill('0') << std::setw(runNumLength) 
		   << lumiSection.hdr.runNumber 
		   << "_" << std::setfill('0') << std::setw(secNumLength) 
		   << lumiSection.hdr.sectionNumber
	 //<< "_" <<  counter 
		   << ".root";
      dbe_->save(tempStreamer.str());
    
   }

   */
}

// ------------ method called once each job just before starting event loop  ------------
void 
HLXMonitor::beginJob(const edm::EventSetup&)
{ 
   HT.SetIP(DistribIP);

   int errorCode = HT.SetPort(listenPort);
   cout << "SetPort: " << errorCode << endl;
   errorCode = HT.SetMode(AquireMode);
   cout << "AquireMode: " << errorCode << endl;
  
   do
   {
      // cout << "Connecting to TCPDistributor" << endl;
      errorCode = HT.Connect();
      if(errorCode != 1)
      {
	 cout << "Attempting to reconnect in " << reconnTime << " seconds." << endl;
	 sleep(reconnTime);
      }
   } while(errorCode != 1);
}

// ------------ method called once each job just after ending the event loop  ------------
void 
HLXMonitor::endJob() 
{
   HT.Disconnect();
}

void HLXMonitor::FillHistoAvg(const LUMI_SECTION & section)
{

   for( unsigned int i = 0; i < NUM_HLX; i++)
   {
      if(section.occupancy[i].hdr.numNibbles != 0)
      {
	 for( unsigned int j = 0; j < NUM_BUNCHES; j++ )
	 {
	    AvgEtSum->Fill(i,section.etSum[i].data[j]);
	
	    AvgOccBelowSet1->Fill(i,section.occupancy[i].data[set1BelowIndex][j]);
	    AvgOccBetweenSet1->Fill(i,section.occupancy[i].data[set1BetweenIndex][j]);
	    AvgOccAboveSet1->Fill(i,section.occupancy[i].data[set1AboveIndex][j]);
	   
	    AvgOccBelowSet2->Fill(i,section.occupancy[i].data[set2BelowIndex][j]);
	    AvgOccBetweenSet2->Fill(i,section.occupancy[i].data[set2BetweenIndex][j]);
	    AvgOccAboveSet2->Fill(i,section.occupancy[i].data[set2AboveIndex][j]);
	
	 }
      }
   }
}

void HLXMonitor::FillHistoBX(const LUMI_SECTION & section)
{

   for( unsigned int i=0; i < NUM_HLX; i++ )
   {
      if(section.occupancy[i].hdr.numNibbles != 0)
      {
	 for( unsigned int j=0; j < NUM_BUNCHES; j++)
	 { 
	    Set1Below[i]->Fill(j,  section.occupancy[i].data[set1BelowIndex][j]);
	    Set1Between[i]->Fill(j,section.occupancy[i].data[set1BetweenIndex][j]);
	    Set1Above[i]->Fill(j,  section.occupancy[i].data[set1AboveIndex][j]);
	    Set2Below[i]->Fill(j,  section.occupancy[i].data[set2BelowIndex][j]);
	    Set2Between[i]->Fill(j,section.occupancy[i].data[set2BetweenIndex][j]);
	    Set2Above[i]->Fill(j,  section.occupancy[i].data[set2AboveIndex][j]);
	    ETSum[i]->Fill(j,      section.etSum[i].data[j]);
	 }
      }
   }
}

void HLXMonitor::FillHistoDist(const LUMI_SECTION & section)
{
   for( unsigned int i=0; i < NUM_HLX; i++)
   {
      if(section.occupancy[i].hdr.numNibbles != 0)
      {
	 for( unsigned int j=0; j < NUM_BUNCHES; j++)
	 { 
	    Set1Below[i]->Fill(  section.occupancy[i].data[set1BelowIndex][j]);
	    Set1Between[i]->Fill(section.occupancy[i].data[set1BetweenIndex][j]);
	    Set1Above[i]->Fill(  section.occupancy[i].data[set1AboveIndex][j]);
	    Set2Below[i]->Fill(  section.occupancy[i].data[set2BelowIndex][j]);
	    Set2Between[i]->Fill(section.occupancy[i].data[set2BetweenIndex][j]);
	    Set2Above[i]->Fill(  section.occupancy[i].data[set2AboveIndex][j]);
	    ETSum[i]->Fill(      section.etSum[i].data[j]);
	 }
      }
   }
}

void HLXMonitor::FillHistoHFCompare(const LUMI_SECTION & section)
{

   for( unsigned int i = 0; i < NUM_HLX; i++ )
   {
      if(section.occupancy[i].hdr.numNibbles != 0)
      {
	 float nActvTwrsSet1 = section.occupancy[i].data[set1AboveIndex][TriggerBX]
	    + section.occupancy[i].data[set1BetweenIndex][TriggerBX]
	    + section.occupancy[i].data[set1BelowIndex][TriggerBX];
      
	 float nActvTwrsSet2 = section.occupancy[i].data[set2AboveIndex][TriggerBX]
	    + section.occupancy[i].data[set2BetweenIndex][TriggerBX]
	    + section.occupancy[i].data[set2BelowIndex][TriggerBX];
      
	 float total = nActvTwrsSet1 + nActvTwrsSet2;

	 if( total != 0)
	 {
	    HFCompareEtSum->Fill( i, section.etSum[i].data[TriggerBX]/total );
	 }

	 if(nActvTwrsSet1 != 0)
	 {
	    float tempData = 
	       section.occupancy[i].data[set1BelowIndex][TriggerBX]/nActvTwrsSet1;
	    HFCompareOccBelowSet1->Fill(i, tempData);
	
	    tempData = 
	       section.occupancy[i].data[set1BetweenIndex][TriggerBX]/nActvTwrsSet1;
	    HFCompareOccBetweenSet1->Fill(i, tempData); 
	
	    tempData = 
	       section.occupancy[i].data[set1AboveIndex][TriggerBX]/nActvTwrsSet1;
	    HFCompareOccAboveSet1->Fill(i, tempData); 
	 }

	 if( nActvTwrsSet2 != 0)
	 {
	    float tempData = 
	       section.occupancy[i].data[set2BelowIndex][TriggerBX]/nActvTwrsSet2;
	    HFCompareOccBelowSet2->Fill(i, tempData);
	
	    tempData = 
	       section.occupancy[i].data[set2BetweenIndex][TriggerBX]/nActvTwrsSet2;
	    HFCompareOccBetweenSet2->Fill(i, tempData); 
	
	    tempData = 
	       section.occupancy[i].data[set2AboveIndex][TriggerBX]/nActvTwrsSet2;
	    HFCompareOccAboveSet2->Fill(i, tempData); 
	 }
      }
   }
}

// void HLXMonitor::FillHistoHistory(const LUMI_SECTION & section)
// {
//   unsigned int i;
//   float ETSumData;
//   float avgOcc[6];

//   for(i=0; i<NUM_HLX; i++){
    
//     AvgOccupancy(section.occupancy[i], avgOcc, NUM_BUNCHES,1);
//     ETSumData = AvgETSum(section.etSum[i],NUM_BUNCHES,1);

//     if(Accumulate==true){
//       avgOcc[set1BelowIndex]   += Set1Below[i]->getBinContent(counter-1);
//       avgOcc[set1BetweenIndex] += Set1Between[i]->getBinContent(counter-1);
//       avgOcc[set1AboveIndex]   += Set1Above[i]->getBinContent(counter-1);
//       avgOcc[set2BelowIndex]   += Set2Below[i]->getBinContent(counter-1);
//       avgOcc[set2BetweenIndex] += Set2Between[i]->getBinContent(counter-1);
//       avgOcc[set2AboveIndex]   += Set2Above[i]->getBinContent(counter-1);
//       ETSumData                += ETSum[i]->getBinContent(counter-1);
//     }

//     Set1Below[i]   ->Fill(counter,avgOcc[set1BelowIndex]);
//     Set1Between[i] ->Fill(counter,avgOcc[set1BetweenIndex]);
//     Set1Above[i]   ->Fill(counter,avgOcc[set1AboveIndex]);
//     Set2Below[i]   ->Fill(counter,avgOcc[set2BelowIndex]);
//     Set2Between[i] ->Fill(counter,avgOcc[set2BetweenIndex]);
//     Set2Above[i]   ->Fill(counter,avgOcc[set2AboveIndex]);
//     ETSum[i]       ->Fill(counter,ETSumData);

//     //setAxisRange doesn't seem to do anything yet.  
//     Set1Below[i]  ->setAxisRange(0,counter+1,1);
//     Set1Between[i]->setAxisRange(0,counter+1,1);
//     Set1Above[i]  ->setAxisRange(0,counter+1,1);
//     Set2Below[i]  ->setAxisRange(0,counter+1,1);
//     Set2Between[i]->setAxisRange(0,counter+1,1);
//     Set2Above[i]  ->setAxisRange(0,counter+1,1);
//     ETSum[i]      ->setAxisRange(0,counter+1,1);
//   }
// }

//define this as a plug-in

DEFINE_FWK_MODULE(HLXMonitor);
