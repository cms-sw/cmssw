#include "DQM/SiStripMonitorHardware/plugins/CnBAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DQM/SiStripMonitorHardware/interface/Fed9UEventAnalyzer.hh"


#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"


// This is the maximum number of histogrammed FEDs
// If the number of FEDs exceeds this limit we have a crash
#define N_MAX_FEDS  (1024)
#define N_MAX_FEDUS (N_MAX_FEDS * 8)

CnBAnalyzer::CnBAnalyzer(const edm::ParameterSet& iConfig) :
  ApveErr(N_MAX_FEDS),          // initialize APVE Error Histogram vector (N_MAX_FEDS FEDS Max)
  ApveErrCount(N_MAX_FEDS),     // initialize the BinCounters vector (for flexibility of presentation, % failure, etc.)
  FeMajApvErr(N_MAX_FEDS),      // initialize APVE Error Histogram vector (N_MAX_FEDS FEDS Max)
  FeWHApv(N_MAX_FEDS),          // initialize APVE Error Histogram vector (N_MAX_FEDS FEDS Max)
  FeLKErr(N_MAX_FEDS),          // initialize APVE Error Histogram vector (N_MAX_FEDS FEDS Max)
  FeSYErr(N_MAX_FEDS),          // initialize APVE Error Histogram vector (N_MAX_FEDS FEDS Max)
  FeRWHErr(N_MAX_FEDS),         // initialize APVE Error Histogram vector (N_MAX_FEDS FEDS Max)
  OosPerFed(N_MAX_FEDS),        // sets the size of the oos per fer per event histogram
  FeMajApvErrCount(N_MAX_FEDS), // initialize the BinCounters vector (for flexibility of presentation, % failure, etc.)
  FsopLong( 2,vector<unsigned long>(8) ),
  FsopShort(8),
  feMajorAddress( N_MAX_FEDS,vector<uint16_t>(8) ), // a grand total of ~ 4000 front end units
  WHError( N_MAX_FEDS,vector<int>(8) ),  // wrong header error
  LKError( N_MAX_FEDS,vector<int>(8) ),  // lock error
  SYError( N_MAX_FEDS,vector<int>(8) ),  // synch error
  RWHError( N_MAX_FEDS,vector<int>(8) ), // RAW wrong header error
  FiberStatusBits( 8, vector<vector<MonitorElement*> >(6,vector<MonitorElement*>(N_MAX_FEDS)) ),//6 histograms per FED FEFPGA for N_MAX_FEDS FED max.
  FiberWHApv( N_MAX_FEDS, vector<MonitorElement*>(8) ),//8 FPGAS for N_MAX_FEDS FEDS
  FiberStatusBitCount( 8, vector<vector<BinCounters*> >(6,vector<BinCounters*>(N_MAX_FEDS)) ),//counter variable for errors/event# precnt.
  feMedianAddr(N_MAX_FEDUS),
  //fenumbers(N_MAX_FEDS)
  firstEvent_(true), // To be removed
  bc(N_MAX_FEDS)    //counts the bits baby//  
{
  // fedEvent_ = new Fed9U::Fed9UDebugEvent(); // new intialization - new = dynamic 
  
  // get hold of back-end interface
  dbe = edm::Service<DaqMonitorBEInterface>().operator->();
  
  //parameters for working with slink and dumping the hex buffer
  swapOn_ = iConfig.getUntrackedParameter<int>("swapOn");
  //   percent_ = iConfig.getUntrackedParameter<int>("percent");
  fileName_ = iConfig.getUntrackedParameter<string>("rootFile");
  //   garb_ = iConfig.getUntrackedParameter<int>("garb");
  //   useCabling_= iConfig.getUntrackedParameter<bool>("UseCabling"); 
  runNumber_ = iConfig.getUntrackedParameter<int>("runNumber");  
  
  N = iConfig.getUntrackedParameter<int>("N");
  
  //Percentage varibale initalizations
  apveErrorPercent = 0;
  
  //Good APV Counter
  badApvCounter = 0;
  
  //nolock
  nolock = 0;

  //out of synch
  oos = 0 ;

  //APV Counter
  goodApvCounter = 0;

  //FE counter
  feEnabledCount = 0;

  //FE enabled
  feEnable = 0;
	
  //actual fe median addr
  medianAddr = 0;
	
  //good fe address counter
  goodFe = 0;

  //percentage matching fe address
  prct = 0;	

  //debug 
  fedCounter = 0;
  //evt counter
  eventCounter = 0;

  // use and throw FEDNumberting object...
  FEDNumbering fedNum;

  // valid FedIds for the tracker
  fedIdBoundaries_ = fedNum.getSiStripFEDIds();

  totalNumberOfFeds_ = fedIdBoundaries_.second - fedIdBoundaries_.first + 1;

}

CnBAnalyzer::~CnBAnalyzer(){

  //delete dynamical vars;
  //   delete fedEvent_;
	
  // go to top directory
  dbe->cd();
  // remove MEs at top directory
  dbe->removeContents(); 
}

void CnBAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){
  int goodAPVs = 0;

  eventCounter;

  using namespace edm;
  using namespace std;

  // To keep track of counting of feds
  vector<int> fednumbers; // Vector of fedIDs for event loop
  vector<int> fen;        // Vector of feEnabled bits for ascertaining address error

  // Retrieve FED raw data ("source" label is now fixed by fwk)
  edm::Handle<FEDRawDataCollection> buffers;
  iEvent.getByType( buffers );

  // Looking for all possible FED ids and placing them
  // in a nice vector

  fedIds_.clear();
  for ( uint16_t ifed = fedIdBoundaries_.first ; ifed <= fedIdBoundaries_.second; ifed++ ) {
    // TODO: remove this 152. Use at least a define!
    if ( buffers->FEDData( static_cast<int>(ifed) ).size() >= 152 ) {
      fedIds_.push_back(ifed);
    }
  }

  LogInfo("FEDBuffer") << "Number of Tracker Fed Buffers:" << fedIds_.size();
  LogInfo("FEDBuffer") << "EVENTNUMB" << iEvent.id().event();


  /**************************/
  /*                        */
  /* Main cycle over FEDs   */
  /*                        */
  /**************************/

  // Counter for total number of enabled / faulty
  // channels in this event
  int total_enabled_channels = 0;
  int total_faulty_channels = 0;

  // Retrieve FED all found fedIds  and iterate through 
  vector<uint16_t>::const_iterator ifed;

  for (ifed = fedIds_.begin() ; ifed != fedIds_.end(); ifed++ ) {
    
    createDetailedFedHistograms((*ifed), runNumber_);
    
    // Retrieve FED raw data for given FED... there it is :)      
    const FEDRawData& input = buffers->FEDData( static_cast<int>(*ifed) );
    Fed9U::u32* data_u32 = 0;
    Fed9U::u32  size_u32 = 0;
    
    data_u32 = reinterpret_cast<Fed9U::u32*>( const_cast<unsigned char*>( input.data() ) );
    size_u32 = static_cast<Fed9U::u32>( input.size() / 4 );
    
    Fed9UEventAnalyzer myEventAnalyzer(fedIdBoundaries_, swapOn_);

    // The Initialize function is true if we have a good
    // non-corrupted tracker buffer
    if (myEventAnalyzer.Initialize(data_u32, size_u32)) {
      
      // The Fed9UErrorCondition structure may cointain all the relevant
      // errors specific to the event
      Fed9UErrorCondition thisFedEventErrs;

      // The actual checkout of the buffer:
      thisFedEventErrs = myEventAnalyzer.Analyze();

      total_enabled_channels += thisFedEventErrs.totalChannels;
      total_faulty_channels  += thisFedEventErrs.problemsSeen;

      // Update counters for FEDs
      fedGenericErrors_->Fill(*ifed, total_faulty_channels);
      if (thisFedEventErrs.internalFreeze) fedFreeze_->Fill(*ifed);
      if (thisFedEventErrs.bxError) fedBx_->Fill(*ifed);
  

      
      // Fill the Front-end failure counters 
      for (unsigned int iFrontEnd=0; iFrontEnd<8; iFrontEnd++) {
	if (thisFedEventErrs.feOverflow[iFrontEnd])
	  feOverFlow_[*ifed]->Fill(iFrontEnd*24+1);
	if (thisFedEventErrs.apvAddressError[iFrontEnd])
	  feAPVAddr_[*ifed]->Fill(iFrontEnd*24+1);
	
      }
      
      // Fill the channel failure counters 
      for (unsigned int iChannel=0; iChannel<96; iChannel++) {
	if (thisFedEventErrs.channel[iChannel]==Fed9UEventAnalyzer::FIBERUNLOCKED)
	  chanErrUnlock_[*ifed]->Fill(iChannel*2+1);
	if (thisFedEventErrs.channel[iChannel]==Fed9UEventAnalyzer::FIBEROUTOFSYNCH)
	  chanErrOOS_[*ifed]->Fill(iChannel*2+1);	
      }      
      
    
      // apv[96*2]
      for (unsigned int iApv=0; iApv<192; iApv++) {
	if (thisFedEventErrs.apv[iApv])
	  badApv_[*ifed]->Fill(iApv+1);
      }

    }
    
  } // end of the for ifed loop


  // TODO: find a better solution to this (and remove the explicit 1000)
  if ( iEvent.id().event()<1000) {
    totalChannels_->Fill( iEvent.id().event(), total_enabled_channels);
    faultyChannels_->Fill( iEvent.id().event(), total_faulty_channels);
  }

} // End of the Event Loop ("analyze function" called once per event)


// ------------ method called once each job just before starting event loop  ------------
void 
CnBAnalyzer::beginJob(const edm::EventSetup& iSetup)
{
  
  // This will be the list of fund (and histogrammed) fedIds.
  
  //   if ( useCabling_ ) {
  
  //     // Retrieve FED cabling
  //     vector<uint16_t> fed_ids;
  
  //     //Retrieve FED ids from cabling map and iterate through 
  //     edm::ESHandle<SiStripFedCabling> cabling;
  //     iSetup.get<SiStripFedCablingRcd>().get( cabling );
  
  //     vector<uint16_t>::const_iterator ifed = cabling->feds().begin();
  //     for ( ; ifed != cabling->feds().end(); ifed++ ) { fed_ids.push_back( *ifed ); }
  
  //     histoNaming( fed_ids, 23 ); // default value set for now
  
  //   } 
  
  createRootFedHistograms(runNumber_);

}

// The following method should be called in place of histoNaming.
// at the job initialization by beginJob()
void CnBAnalyzer::createRootFedHistograms( const int& runNumber ) {

  uint16_t fedId;

  dbe->setCurrentFolder("");


  // This Histogram will be filled with 
  // problemsSeen / totalChannels
  fedGenericErrors_ = dbe->book1D( "Fed Generic Errors","Fed Generic Errors vs. FED #",
				   totalNumberOfFeds_,
				   fedIdBoundaries_.first  - 0.5,
				   fedIdBoundaries_.second + 0.5 );
  
  
  // bool internalFreeze
  fedFreeze_ = dbe->book1D( "Fed Freeze","Fed Freeze vs. FED #",
			    totalNumberOfFeds_,
			    fedIdBoundaries_.first  - 0.5,
			    fedIdBoundaries_.second + 0.5 );
  
  // bool bxError
  fedBx_ = dbe->book1D( "Fed Bx Error","Fed Bx Error vs. FED #",
			totalNumberOfFeds_,
			fedIdBoundaries_.first  - 0.5,
			fedIdBoundaries_.second + 0.5 );

  // Trend plots:
  totalChannels_  = dbe->book1D( "Total channels vs. Event",
				 "Total channels vs. Event for all FEDs",
				 1001, 0.5, 1000.5);

  faultyChannels_ = dbe->book1D( "Faulty channels vs. Event",
				 "Faulty channels vs. Event for all FEDs",
				 1001, 0.5, 1000.5);

  // TODO: find a better solution for the trend plot. Better than booking only the first 1001 events...!


  // Previous plots have a bin for every *possible* Tracker FED
  // The directory specific to each FED is build on-demand (data driven)
  // by means of the following method:
  // createDetailedFedHistograms(fedId, runNumber);


  // NOT TRUE - NOT TRUE - NOT TRUE - NOT TRUE - NOT TRUE
  // NOT TRUE - NOT TRUE - NOT TRUE - NOT TRUE - NOT TRUE
  // Directories and data structures are created for all ppossible FEDs
  // It will be up to the client to look only at the sensible plots.
  // NOT TRUE - NOT TRUE - NOT TRUE - NOT TRUE - NOT TRUE
  // NOT TRUE - NOT TRUE - NOT TRUE - NOT TRUE - NOT TRUE

  // The following lines are unused.
  //   for (fedId = fedIdBoundaries_.first; fedId<=fedIdBoundaries_.second; fedId++ ) {
  //     createDetailedFedHistograms(fedId, runNumber);
  //   }
}

void CnBAnalyzer::createDetailedFedHistograms( const uint16_t& fed_id, const int& runNumber ) {

  std::map<int, bool>::iterator itFeds;

  itFeds=foundFeds_.find(fed_id);

  if (itFeds==foundFeds_.end()) {
    foundFeds_[fed_id]=true;
    
    stringstream  fedNumber;
    fedNumber << fed_id;
    string f = "FED #"; 
    
    dbe->setCurrentFolder(f+fedNumber.str());
    
    // All the following histograms are such that thay can be plot together
    // In fact the boundaries of the plots are 1, 192 (or actually 0.5, 192,5)
    // They have a different binning, though, which reflect the granularity of the system.
    
    // When filling these plots one can access directly the bin (for example feOverFlow, bin number 8
    // to address the front-end unit number 8) or one can make use of the x axis, which correspond
    // to the APV index (1, 192).
    
    
    //   bool feOverflow[8];
    feOverFlow_[fed_id]     = dbe->book1D( "FedUnit Overflow","FedUnit Overflow for FED #"+fedNumber.str(),
					   8, 0.5, 192.5 );
    
    //   bool apvAddressError[8];
    feAPVAddr_[fed_id]      = dbe->book1D( "APV Address error","FedUnit APV Address error for FED #"+fedNumber.str(),
					   8, 0.5, 192.5 );
    
    // Channel[96]
    chanErrUnlock_[fed_id]  = dbe->book1D( "Unlock error", "Unlocked Fiber error for FED #"+fedNumber.str(),
					   96, 0.5, 192.5);
    chanErrOOS_[fed_id]     = dbe->book1D( "OOS error", "OutOfSynch Fiber error for FED #"+fedNumber.str(),
					   96, 0.5, 192.5);
    
    // apv[96*2]
    badApv_[fed_id]         = dbe->book1D( "Bad APV error", "Bad APV error for FED #"+fedNumber.str(),
					   192, 0.5, 192.5);


  } // Otherwise we have nothing to do
  
}


// OBSOLETE
// ------------ method called once each job just before starting event loop  ------------
void 
CnBAnalyzer::histoNaming( const vector<uint16_t>& fed_ids, const int& runNumber ) {
  std::cout << "fedIds: ";
  for ( vector<uint16_t>::const_iterator i = fed_ids.begin(); i != fed_ids.end(); i++ ) {
    std::cout << (*i) << ' ';
  }
  std::cout << std::endl;
	
  int fedCounter2 = 0;
  int runNo = runNumber;
  stringstream ss;
  stringstream ssi;
  stringstream ssii;
  string f = "FED #"; 

  //histogram labels for the status bits
  vector<string> statusBits(6);
  statusBits[0]="APVerrorB<APV0>";
  statusBits[1]="wrong_headerB<APV0>";
  statusBits[2]="APVerrorB<APV1>";
  statusBits[3]="wrong_headerB<APV1>";
  statusBits[4]="out_of_synchB";
  statusBits[5]="lock";
 
  vector<uint16_t>::const_iterator ifed = fed_ids.begin();
  for ( ; ifed != fed_ids.end(); ifed++ ) {
	  
    fedCounter2++; //total number of feds in cabling map

    ss<< *ifed;

    //Monitoring Hisotgram Declarations and Setup
    //-------------------------------------------------
    //APV Address Error Histograms
    dbe->setCurrentFolder( f+ss.str()+"/Errors per FPGA" );
    ApveErr[*ifed] = dbe->book1D( "APVE Address Error","APVE Address Error FED#"+ss.str() , 8, 0, 8 );
    FeMajApvErr[*ifed] = dbe->book1D( "FE Majority Address Error","FE Majority Address Error FED#"+ss.str() , 8, 0, 8 );
    FeWHApv[*ifed] = dbe->book1D( "Wrong Header Apv Error","APV Wrong Header Error per FPGA FED#"+ss.str() , 8, 0, 8 );
    FeLKErr[*ifed] = dbe->book1D( "Lock Error","Lock Error per FPGA FED#"+ss.str() , 8, 0, 8 );
    FeSYErr[*ifed] = dbe->book1D( "Synch Error","Synch Error per FPGA FED#"+ss.str() , 8, 0, 8 );
    FeRWHErr[*ifed] = dbe->book1D( "RAW Wrong Header Apv Error"," RAW APV Wrong Header Error per FPGA FED#"+ss.str() , 8, 0, 8 );
    ApveErrCount[*ifed] = new BinCounters;
    FeMajApvErrCount[*ifed] = new BinCounters;
    for(int i = 0; i < 8; i++){
      ssi << i+1;
      ApveErr[*ifed]->setBinLabel( i+1,"FPGA #"+ssi.str() );
      FeMajApvErr[*ifed]->setBinLabel( i+1,"FPGA #"+ssi.str() );
      FeWHApv[*ifed]->setBinLabel( i+1,"FPGA #"+ssi.str() );
      ssi.str(" ");
    }
    dbe->setCurrentFolder( f+ss.str()+"/Out of Synch Per Event" );
    OosPerFed[*ifed] = dbe->book1D("Oos per FED ", "oos for  FED #"+ss.str(),1100, 0, 1100); 
    //FE FPGA Status Bit Histograms
    for(int i = 0; i < 8; i++){
      ssi << i+1;
      dbe->setCurrentFolder( f+ss.str()+"/FPGA #"+ssi.str()+" WH Errors" );
      FiberWHApv[*ifed][i] = dbe->book1D( "WH GOOD APV" ," APV Wrong Header Errors per Fiber for FED #"+ss.str()
					  +" FPGA #"+ssi.str() , 12, 0, 12 );
      errors[*ifed][i] = dbe->book1D( "anyError","Any APV or sync error per fiber for FED #"+ss.str()+" FPGA #"+ssi.str() , 12, 0, 12 );

      dbe->setCurrentFolder( f+ss.str()+"/FPGA #"+ssi.str()+" Fiber Status Bits" );
      for(int j = 0; j < 6; j++){
	FiberStatusBits[i][j][*ifed] = dbe->book1D( statusBits[j] ,statusBits[j]+" for FED#"+ss.str()
						    +" FPGA #"+ssi.str() , 12, 0, 12 );
	FiberStatusBitCount[i][j][*ifed] = new BinCounters;
	for(int k = 0; k < 12; k++){
	  ssii << k+1;
	  FiberWHApv[*ifed][i]->setBinLabel( k+1, "Fiber #"+ssii.str() );
	  errors[*ifed][i]->setBinLabel( k+1, "Fiber #"+ssii.str() );
	  FiberStatusBits[i][j][*ifed]->setBinLabel( k+1, "Fiber #"+ssii.str() );
	  ssii.str(" ");
	}
      }
      ssi.str("");
    }		

    //---------------------------------------------------------------------------------------------------------------------------
    ss.str(" "); // Clear fed id ss Stream
  }// FED ID For Loop terminus

  //GLobal ( Per Event Histograms )

  ss<<runNo;	

  dbe->setCurrentFolder( "Global Add Consist Check Run No."+ss.str() );
  AddCheck0 = dbe->book1D( "FE Add Consist Check Run #"+ss.str(),"FED Consist Check Run #"+ss.str(), (fedCounter2 * 8), -0.5, ( (fedCounter2 * 8)+0.5) );

		
  dbe->setCurrentFolder( "Fraction of FE in Synch Per Event Run #"+ss.str() );
  AddConstPerEvent = dbe->book1D( "Percentage Synch for Run #"+ss.str(),"Prct. in Synchrony Run #"+ss.str(), 1000, 0, 1000 );

  ApvAddConstPerEvent = dbe->book1D( "Percentage APV Synch for Run #"+ss.str(),"Prct. APV in Synchrony Run #"+ss.str(), 1000, 0, 1000 );
  ApvAddConstPerEvent1 = dbe->book1D( "Percentage1 APV Synch for Run #"+ss.str(),"Prct1. APV in Synchrony Run #"+ss.str(), 1000, 0, 1000 );
  ApvAddConstPerEvent2 = dbe->book1D( "PercentageLK APV Synch for Run #"+ss.str(),"PrctLK. APV in Synchrony Run #"+ss.str(), 1000, 0, 1000 );

	
  NoLock = dbe->book1D( "NoLock for Run #"+ss.str(),"Unlocked Fibers per Event Run #"+ss.str(), 1000, 0, 1000 );
  NoSynch = dbe->book1D( "NoSynch for Run #"+ss.str(),"Out of Synch Fibers per Event Run #"+ss.str(), 1000, 0, 1000 );

  BadHead = dbe->book1D( "RAW Wrong Header Run #"+ss.str(),"RAW Wrong Header Errors per Event Run #"+ss.str(), 1000, 0, 1000 );


  dbe->setCurrentFolder( "Cumulative Number of Address Errors Per FED Run #"+ss.str() );
  CumNumber = dbe->book1D("Cumulative FE Errors per FED Run #"+ss.str(), "cumulative FE errors for feds Run #"+ss.str(),
			  fedCounter2, 0, fedCounter2); //set bin label in FED loop in the main program...			
	    

  CumNumber1 = dbe->book1D("Cumulative Number of APV Wrong Header Errors per FED Run #"+ss.str(), "Total APV Wrong Header Errors per FED for  Run #"+ss.str(),
			   fedCounter2, 0, fedCounter2); //set bin label in FED loop in the main program...
  CumNumber2 = dbe->book1D("Cumulative Number of Lock Errors per FED Run #"+ss.str(), "Total Lock Errors per FED for  Run #"+ss.str(),
			   fedCounter2, 0, fedCounter2); //set bin label in FED loop in the main program...
  CumNumber3 = dbe->book1D("Cumulative Number of Synch Errors per FED Run #"+ss.str(), "Total Synch Errors per FED for  Run #"+ss.str(),
			   fedCounter2, 0, fedCounter2); //set bin label in FED loop in the main program...
  CumNumber4 = dbe->book1D("Cumulative Number of RAW APV Header Errors per FED Run #"+ss.str(), "Total RAW APV Header Errors per FED for  Run #"+ss.str(),
			   fedCounter2, 0, fedCounter2); //set bin label in FED loop in the main program...





  //OosPerFed = dbe->book1D("Oos per FED ", "oos for  #"+ss.str(),fedCounter2, 0, fedCounter2); //set bin label in FED


  //dbe->setCurrentFolder( "Out of Synch with FEDS and Event # FED Run #"+ss.str() );
  //oosFedEvent=dbe->book2D("OOS per FED per Evt.","OOS per FED per Event", 1000, 0, 1000,fedCounter2, 0, fedCounter2); 
  dbe->setCurrentFolder("APVs on Good Fibers" + ss.str());
  goodAPVsPerEvent_ = dbe->book1D("goodAPVsPerEvent" + ss.str(),"Working APVs as a percentage working fibers"+ss.str(),1000,0,1000);

  std::cout<<"FEDCTR2"<<fedCounter2<<std::endl;
	
  ss.str(" ");

} //

// ------------ method called once each job just after ending the event loop  ------------
void 
CnBAnalyzer::endJob() {

  for (unsigned int i = 0; i < ApveErrCount.size(); i++ ) {
    delete ApveErrCount[i];
  }

  dbe->showDirStructure();
  dbe->save(fileName_);
	

  //   int sz = apvPrct.size();

  //   std::cout<<" SIZZY "<<sz<<std::endl;

  //   sort(apvPrct.begin(), apvPrct.end());
  //   std::cout<<" MIZZY "<<apvPrct[sz - 1]<<std::endl;
  //   std::cout<<" LIZZY "<<apvPrct[0]<<std::endl;

  //   map<double,int> m1;
  //   for(pi = apvPrct.begin(); pi != apvPrct.end(); pi++)
  //     {
  //       m1[*pi]++;
  //     }
  //   int prevHighCount = 0;
  //   float answer =0 ;
  //   for(std::map<double,int>::iterator i = m1.begin(); i != m1.end(); i++){
  //     if((*i).second > prevHighCount){
  //       prevHighCount = (*i).second;
  //       answer = (*i).first;
  //     }
  //   }



  //   std::cout << " MOOZY " << answer << std::endl;



}



