#include "DQM/SiStripMonitorHardware/plugins/CnBAnalyzer.h"

CnBAnalyzer::CnBAnalyzer(const edm::ParameterSet& iConfig) :
  ApveErr(2000),          // initialze APVE Error Histogram vector (2000 FEDS Max)
  ApveErrCount(2000),     // initialize the BinCounters vector (for flexibility of presentation, % failure, etc.)
  FeMajApvErr(2000),      // initialze APVE Error Histogram vector (2000 FEDS Max)
  FeWHApv(2000),          // initialze APVE Error Histogram vector (2000 FEDS Max)
  FeLKErr(2000),          // initialze APVE Error Histogram vector (2000 FEDS Max)
  FeSYErr(2000),          // initialze APVE Error Histogram vector (2000 FEDS Max)
  FeRWHErr(2000),         // initialze APVE Error Histogram vector (2000 FEDS Max)
  OosPerFed(2000),        // sets the size of the oos per fer per event histogram
  FeMajApvErrCount(2000), // initialize the BinCounters vector (for flexibility of presentation, % failure, etc.)
  FsopLong( 2,vector<unsigned long>(8) ),
  FsopShort(8),
  feMajorAddress( 2000,vector<uint16_t>(8) ), // a grand total of ~ 4000 front end units
  WHError( 2000,vector<int>(8) ),  // wrong header error
  LKError( 2000,vector<int>(8) ),  // lock error
  SYError( 2000,vector<int>(8) ),  // synch error
  RWHError( 2000,vector<int>(8) ), // RAW wrong header error
  FiberStatusBits( 8, vector<vector<MonitorElement*> >(6,vector<MonitorElement*>(2000)) ),//6 histograms per FED FEFPGA for 2000 FED max.
  FiberWHApv( 2000, vector<MonitorElement*>(8) ),//8 FPGAS for 2000 FEDS
  FiberStatusBitCount( 8, vector<vector<BinCounters*> >(6,vector<BinCounters*>(2000)) ),//counter variable for errors/event# precnt.
  feMedianAddr(4000),
  //fenumbers(2000)
  fedIds_(),
  firstEvent_(true),
  bc(2000),    //counts the bits baby
  errors( 2000, vector<MonitorElement*>(8) )
  //useCabling_( iConfig.getUntrackedParameter<bool>("UseCabling",false) )
{
  fedEvent_ = new Fed9U::Fed9UDebugEvent(); // new intialization - new = dynamic 
  
  // get hold of back-end interfaec
  dbe = edm::Service<DaqMonitorBEInterface>().operator->();
  
  //parameters for working with slink and dumping the hex buffer
  swapOn_ = iConfig.getUntrackedParameter<int>("swapOn");
  dump_ = iConfig.getUntrackedParameter<int>("dump");
  wordNumber_ = iConfig.getUntrackedParameter<int>("wordNumber");
  percent_ = iConfig.getUntrackedParameter<int>("percent");
  fileName_ = iConfig.getUntrackedParameter<string>("rootFile");
  garb_ = iConfig.getUntrackedParameter<int>("garb");
  useCabling_= iConfig.getUntrackedParameter<bool>("UseCabling"); 
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

}

CnBAnalyzer::~CnBAnalyzer(){

  //delete dynamical vars;
  delete fedEvent_;
	
  // go to top directory
  dbe->cd();
  // remove MEs at top directory
  dbe->removeContents(); 
}

void CnBAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){
  int goodFibers = 0;
  int goodAPVs = 0;
  int nFi = 0;

  eventCounter;

  using namespace edm;
  using namespace std;

  //	std::cout<<setfill('0'); //set for formatting

  //Fed9U::Fed9UDebugEvent::EnableDebug(true);

  //theres got to be a better way
  int feCtr = 0; //counts number of fe fpgas	
  int tmpCtr = 0; //
  BinCounters inv8; //invokes a function that reverses the bit order of a number

  //to keep track of counting of feds
  vector<int> fednumbers; //vector of fed ids for event loop
  vector<int> fen; //vector of feEnabled bits for ascertaining address error

  stringstream ss; 
  stringstream ssi; // for the apv err part
  // Retrieve FED raw data ("source" label is now fixed by fwk)
  edm::Handle<FEDRawDataCollection> buffers;
  iEvent.getByType( buffers ); 

  // Retrieve FED cabling
  if ( firstEvent_ ) {
    firstEvent_ = false;

    // build fed ids vector
    fedIds_.clear();
    if ( useCabling_ ) {
      edm::ESHandle<SiStripFedCabling> cabling;
      iSetup.get<SiStripFedCablingRcd>().get( cabling );
      vector<uint16_t>::const_iterator ifed = cabling->feds().begin();
      for ( ; ifed != cabling->feds().end(); ifed++ ) { fedIds_.push_back( *ifed ); }
    } else { 
      for ( uint16_t ifed = 0; ifed < 1023; ifed++ ) {
	if ( buffers->FEDData( static_cast<int>(ifed) ).size() >= 152 ) {
	
	  fedIds_.push_back(ifed);
	}
      }
      // create histos
      histoNaming( fedIds_ , runNumber_);
    }
      
  }
    
  // Retrieve FED ids from cabling map and iterate through 
  vector<uint16_t>::const_iterator ifed = fedIds_.begin();
  uint16_t total_enabled_channels = 0;

  std::cout<< "FEDIDSIZE"<<fedIds_.size()<<std::endl;

  std::cout<< "EVENTNUMB"<<iEvent.id().event()<<std::endl;


  //XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
  for ( ; ifed != fedIds_.end(); ifed++ ) {

    // Retrieve FED raw data for given FED ..there is is :)      
    const FEDRawData& input = buffers->FEDData( static_cast<int>(*ifed) );
    Fed9U::u32* data_u32 = 0;
    Fed9U::u32  size_u32 = 0;

    data_u32 = reinterpret_cast<Fed9U::u32*>( const_cast<unsigned char*>( input.data() ) );
    size_u32 = static_cast<Fed9U::u32>( input.size() / 4 ); 
	
    if(data_u32 == NULL){ std::cerr<<"data_u32 is NULL !!"<<std::endl; continue; }
	
    //ignores buffers of zero size (container (fed ID) is present but contains nothing) 
    if(!size_u32) { /*ifed++;*/ continue;} // loops back if non zero, increments ifed iterator to next container 

    //adjusts the buffer pointers for the DAQ header and trailer present when FRLs are running
    //additonally preforms "flipping" of the bytes in the buffer
    if(swapOn_){ 
	
      Fed9U::u32 temp1,temp2;
		
      //32 bit word swapping for the real FED buffers	
      for(unsigned int i = 0; i < (size_u32 - 1); i+=2){
	
	temp1 = *(data_u32+i);
	temp2 = *(data_u32+i+1);
	*(data_u32+i) = temp2;
	*(data_u32+i+1) = temp1;
		
      }
	
      //dumps a specified number of 32 bit words to the screen prior ot initalization
      if(dump_){
			
	std::cerr<<"BUFFERD FED NUMBER "<<dec<<*ifed<<"For EVT "<<iEvent.id().event()<<std::endl;
	
	for(int i = 0; i<wordNumber_; i++) // prints out the specified number of 32 bit words
	  {
	    setiosflags(ios::left);
	    std::cerr<<setw(2);
	    std::cerr<<"64 Bit Word # "<<" "<<dec<<i<<" "
		<<hex<<(data_u32[2*i] & 0xFFFFFFFF)<<" "
		<<(data_u32[(2*i + 1)] & 0xFFFFFFFF)<<" "
		<<std::endl;
	    //std::cerr<<"data_u32 word # "<<" "<<i<<" "<<hex<<data_u32[i]<<std::endl;
	  }
				
      }
      if (!data_u32 || !size_u32 )continue;
      try{ 
        fedEvent_->Init( data_u32, 0, size_u32 ); // initialize the fedEvent with offset for slink
      } catch(...) {
        continue;
      }
      total_enabled_channels += fedEvent_->totalChannels(); 
		
      /*	
      //Double Checksss......
      //dumps a specified number of 32 bit words to the screen prior ot initalization
      for(int i = 0; i<wordNumber_; i++) // prints out the specified number of 32 bit words
      {
      setiosflags(ios::left);
      std::cerr<<setw(2);
      std::cerr<<"data_u32 word # "<<" "<<i<<" "<<hex<<data_u32[i]<<std::endl;
      }
      */
		
    } else {

      //dumps a specified number of 32 bit words to the screen prior ot initalization
      if(dump_){

	std::cerr<<dec<<"BUFFER DUMP FOR FED NUMBER "<<*ifed<<std::endl;

	for(int i = 0; i<wordNumber_; i++) // prints out the specified number of 32 bit words
	  {
	    setiosflags(ios::left);
	    std::cerr<<setw(2);
	    std::cerr<<"64 Bit Word # "<<" "<<dec<<i<<" "
		<<hex<<(data_u32[2*i] & 0xFFFFFFFF)<<" "
		<<(data_u32[(2*i + 1)] & 0xFFFFFFFF)<<" "
		<<std::endl;
	    //std::cerr<<"data_u32 word # "<<" "<<i<<" "<<hex<<data_u32[i]<<std::endl;
	  }

      }
	
      if (!data_u32 || !size_u32 )continue;
      try{
        fedEvent_->Init( data_u32, 0, size_u32 ); // initialize the fedEvent with offset for slink
      } catch(...) {
        continue;
      }
    }

    /*
    //listed in order of appearance in event formats paper : reads out the entire first line on pg. 6
    edm::LogInfo("FedId ") << *ifed;
    edm::LogInfo("First Byte Reserved -> $ED ") << static_cast<uint16_t>(fedEvent_->getSpecialFirstByte() );
    edm::LogInfo("Hdr Format ") << static_cast<uint16_t>(fedEvent_->getSpecialHeaderFormat() );
    edm::LogInfo("Tracker Event Type ") << static_cast<uint16_t>(fedEvent_->getSpecialTrackerEventType() );
    edm::LogInfo("APVE Address " ) << static_cast<uint16_t>(fedEvent_->getSpecialApvEmulatorAddress() );
    edm::LogInfo("APV Address Error ") << static_cast<uint16_t>(fedEvent_->getSpecialApvAddressError() );


    std::cout<<" INVERSIONTEST "<<inv8.invert8(221)<<std::endl;

    std::cout<<dec<<"FEE FOR FED # "<<*ifed<<" is "<<static_cast<uint16_t>(fedEvent_->getSpecialFeEnableReg() )<<std::endl;
    edm::LogInfo("FE Enable Register ") << static_cast<uint16_t>(fedEvent_->getSpecialFeEnableReg() );
    */

    //	feEnable = static_cast<uint16_t>(fedEvent_->getSpecialFeEnableReg() ) ;
    /*

    edm::LogInfo("FE Overflow Register (fixed at 0) ") << static_cast<uint16_t>(fedEvent_->getSpecialFeOverflowReg() );


    edm::LogInfo("FE FED Status Register ") << static_cast<uint16_t>(fedEvent_->getSpecialFedStatusRegister() );

    //testing for 4/13/07
    */
    FsopLong[0][7]= static_cast<unsigned long>(fedEvent_->getFSOP_8_1() );
    FsopLong[1][7]= static_cast<unsigned long>(fedEvent_->getFSOP_8_2() );
    FsopShort[7]= static_cast<uint16_t>(fedEvent_->getFSOP_8_3() );

    //      	edm::LogInfo("FE FSOP 8_1") << static_cast<unsigned long>(fedEvent_->getFSOP_8_1() );
    //      	edm::LogInfo("FE FSOP 8_2") << static_cast<unsigned long>(fedEvent_->getFSOP_8_2() ); 
    //	edm::LogInfo("FE FSOP 8_3") << static_cast<uint16_t>(fedEvent_->getFSOP_8_3() );
	
    std::cout<<"LAST"<<static_cast<uint16_t>(fedEvent_->getFSOP_8_3() )<<std::endl;

    edm::LogInfo("FE 8 LENGTH") << static_cast<uint16_t>(fedEvent_->getFLEN_8() );
    edm::LogInfo("BESR") << static_cast<unsigned long>(fedEvent_->getBESR() );
    //edm::LogInfo("FE BESR2") << static_cast<unsigned long>(fedEvent_->getBESR_2() );
	
    FsopLong[0][6]= static_cast<unsigned long>(fedEvent_->getFSOP_7_1() );
    FsopLong[1][6]= static_cast<unsigned long>(fedEvent_->getFSOP_7_2() );
    FsopShort[6]= static_cast<uint16_t>(fedEvent_->getFSOP_7_3() );


    /*

    edm::LogInfo("FE FSOP 7_1") << static_cast<unsigned long>(fedEvent_->getFSOP_7_1() );
    edm::LogInfo("FE FSOP 7_2") << static_cast<unsigned long>(fedEvent_->getFSOP_7_2() ); 
    edm::LogInfo("FE FSOP 7_3") << static_cast<uint16_t>(fedEvent_->getFSOP_7_3() );

    edm::LogInfo("FE 7 LENGTH") << static_cast<uint16_t>(fedEvent_->getFLEN_7() );
    edm::LogInfo("RESERVED 5") << static_cast<unsigned long>(fedEvent_->getRES_5() );
    */
    FsopLong[0][5]= static_cast<unsigned long>(fedEvent_->getFSOP_6_1() );
    FsopLong[1][5]= static_cast<unsigned long>(fedEvent_->getFSOP_6_2() );
    FsopShort[5]= static_cast<uint16_t>(fedEvent_->getFSOP_6_3() );
    	
    /*
      edm::LogInfo("FE FSOP 6_1") << static_cast<unsigned long>(fedEvent_->getFSOP_6_1() );
      edm::LogInfo("FE FSOP 6_2") << static_cast<unsigned long>(fedEvent_->getFSOP_6_2() ); 
      edm::LogInfo("FE FSOP 6_3") << static_cast<uint16_t>(fedEvent_->getFSOP_6_3() );


      edm::LogInfo("FE 6 LENGTH") << static_cast<uint16_t>(fedEvent_->getFLEN_6() );
      edm::LogInfo("RESERVED 4") << static_cast<unsigned long>(fedEvent_->getRES_4() );
    */	
    FsopLong[0][4]= static_cast<unsigned long>(fedEvent_->getFSOP_5_1() );
    FsopLong[1][4]= static_cast<unsigned long>(fedEvent_->getFSOP_5_2() );
    FsopShort[4]= static_cast<uint16_t>(fedEvent_->getFSOP_5_3() );
 
    /*
      edm::LogInfo("FE FSOP 5_1") << static_cast<unsigned long>(fedEvent_->getFSOP_5_1() );
      edm::LogInfo("FE FSOP 5_2") << static_cast<unsigned long>(fedEvent_->getFSOP_5_2() ); 
      edm::LogInfo("FE FSOP 5_3") << static_cast<uint16_t>(fedEvent_->getFSOP_5_3() );


      edm::LogInfo("FE 5 LENGTH") << static_cast<uint16_t>(fedEvent_->getFLEN_5() );
      edm::LogInfo("RESERVED 3") << static_cast<unsigned long>(fedEvent_->getRES_3() );
    */
    FsopLong[0][3]= static_cast<unsigned long>(fedEvent_->getFSOP_4_1() );
    FsopLong[1][3]= static_cast<unsigned long>(fedEvent_->getFSOP_4_2() );
    FsopShort[3]= static_cast<uint16_t>(fedEvent_->getFSOP_4_3() );

    /*  
    	edm::LogInfo("FE FSOP 4_1") << static_cast<unsigned long>(fedEvent_->getFSOP_4_1() );
      	edm::LogInfo("FE FSOP 4_2") << static_cast<unsigned long>(fedEvent_->getFSOP_4_2() ); 
	edm::LogInfo("FE FSOP 4_3") << static_cast<uint16_t>(fedEvent_->getFSOP_4_3() );


	edm::LogInfo("FE 4 LENGTH") << static_cast<uint16_t>(fedEvent_->getFLEN_4() );
      	edm::LogInfo("RESERVED 2") << static_cast<unsigned long>(fedEvent_->getRES_2() );
    */
    FsopLong[0][2]= static_cast<unsigned long>(fedEvent_->getFSOP_3_1() );
    FsopLong[1][2]= static_cast<unsigned long>(fedEvent_->getFSOP_3_2() );
    FsopShort[2]= static_cast<uint16_t>(fedEvent_->getFSOP_3_3() );
  
    /*
      edm::LogInfo("FE FSOP 3_1") << static_cast<unsigned long>(fedEvent_->getFSOP_3_1() );
      edm::LogInfo("FE FSOP 3_2") << static_cast<unsigned long>(fedEvent_->getFSOP_3_2() ); 
      edm::LogInfo("FE FSOP 3_3") << static_cast<uint16_t>(fedEvent_->getFSOP_3_3() );


      edm::LogInfo("FE 3 LENGTH") << static_cast<uint16_t>(fedEvent_->getFLEN_3() );
      edm::LogInfo("RESERVED 1") << static_cast<unsigned long>(fedEvent_->getRES_1() );
    */
    FsopLong[0][1]= static_cast<unsigned long>(fedEvent_->getFSOP_2_1() );
    FsopLong[1][1]= static_cast<unsigned long>(fedEvent_->getFSOP_2_2() );
    FsopShort[1]= static_cast<uint16_t>(fedEvent_->getFSOP_2_3() );
      	
    /*
      edm::LogInfo("FE FSOP 2_1") << static_cast<unsigned long>(fedEvent_->getFSOP_2_1() );
      edm::LogInfo("FE FSOP 2_2") << static_cast<unsigned long>(fedEvent_->getFSOP_2_2() ); 
      edm::LogInfo("FE FSOP 2_3") << static_cast<uint16_t>(fedEvent_->getFSOP_2_3() );


      edm::LogInfo("FE 2 LENGTH") << static_cast<uint16_t>(fedEvent_->getFLEN_2() );
      edm::LogInfo("DAQ REGISTER 2") << static_cast<unsigned long>(fedEvent_->getDAQ_2() );
    */
    FsopLong[0][0]= static_cast<unsigned long>(fedEvent_->getFSOP_1_1() );
    FsopLong[1][0]= static_cast<unsigned long>(fedEvent_->getFSOP_1_2() );
    FsopShort[0]= static_cast<uint16_t>(fedEvent_->getFSOP_1_3() );
  
    /*
      edm::LogInfo("FE FSOP 1_1") << static_cast<unsigned long>(fedEvent_->getFSOP_1_1() );
      edm::LogInfo("FE FSOP 1_2") << static_cast<unsigned long>(fedEvent_->getFSOP_1_2() ); 
      edm::LogInfo("FE FSOP 1_3") << static_cast<uint16_t>(fedEvent_->getFSOP_1_3() );


      edm::LogInfo("FE 1 LENGTH") << static_cast<uint16_t>(fedEvent_->getFLEN_1() );
      edm::LogInfo("DAQ REGISTER 1") << static_cast<unsigned long>(fedEvent_->getDAQ_1() );
	
    */
    if( garb_ ){	
      feEnable = static_cast<uint16_t>(fedEvent_->getSpecialFeEnableReg() ) ;
      fen.push_back( static_cast<uint16_t>(fedEvent_->getSpecialFeEnableReg() ) );
    }
    else{
      feEnable = inv8.invert8(static_cast<uint16_t>(fedEvent_->getSpecialFeEnableReg() ) );
      fen.push_back( inv8.invert8(static_cast<uint16_t>(fedEvent_->getSpecialFeEnableReg() ) )  );
    }
    //the DQM Plotting-mapping portion++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++	
    string f = "FED #";
    //keeps track of feds
    ss << *ifed;
    fedCounter++; //counter to keep track of the total number of feds in loop
    fednumbers.push_back(*ifed);

    //for 2D plot bin setting ... what is the deal with the naming on this ?
    //oosFedEvent->setBinLabel(fedCounter, ss.str(),2);


    //APV Address Error Histogram Filling	
    //Count the number of errors on an Event by Event basis	
    ApveErrCount[*ifed]->setBinCounters( inv8.invert8( static_cast<uint16_t>(fedEvent_->getSpecialApvAddressError()) ) );
	
    //APV Error Histo filled every N events and updated as a percent
    if(percent_){
	
      if(iEvent.id().event() % N == 0){
	for(int i = 0; i<8 ; i++){
	  if( 0x1 & (feEnable >> i) ){
	    apveErrorPercent = ((float)ApveErrCount[*ifed]->getBinCounters(i) / (float)iEvent.id().event());
	    ApveErr[*ifed]->setBinContent(i+1, apveErrorPercent);
	  }
	}
      }
				
    }
    //APV Error Histo per Event-------------------------------------------------------------------------------------------
    else{	
			
      for(int i = 0; i < 8; i++){
	if( 0x1 & (feEnable >> i) ){
	  if( (0x1 & ( inv8.invert8( static_cast<uint16_t>(fedEvent_->getSpecialApvAddressError())  ) >> i)) == 0){
	    ApveErr[*ifed]->Fill(i) ;
	  }
	}
      }

    }
    //------------------------------------------------------------------------------------------------------------

    //Fiber Status Histos per Event  


	
    //XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    //takes into account frame synch out packet bits 0 - 31

    int problemsSeen = 0;

    for(int fpga = 0; fpga < 8 ; fpga++){
      if( 0x1 & (feEnable >> fpga) ){
	feCtr++; //increment the front end std::couter var

	int feErr = 0;
	int lkErr = 0;
	int syErr = 0;
	int rfeErr = 0;

	// Nicks logic for getting subset in synch and locked---------..------------------------------------------
	for (int fi=0; fi<12; fi++) {
	  bool APV1Error = !getBit(fi*6,FsopLong[0][fpga],FsopLong[1][fpga],FsopShort[fpga]);
	  bool APV1WrongAdd = !getBit(fi*6+1,FsopLong[0][fpga],FsopLong[1][fpga],FsopShort[fpga]);
	  bool APV2Error = !getBit(fi*6+2,FsopLong[0][fpga],FsopLong[1][fpga],FsopShort[fpga]);
	  bool APV2WrongAdd = !getBit(fi*6+3,FsopLong[0][fpga],FsopLong[1][fpga],FsopShort[fpga]);
	  bool outOfSync = !getBit(fi*6+4,FsopLong[0][fpga],FsopLong[1][fpga],FsopShort[fpga]);
	  bool unlocked = !getBit(fi*6+5,FsopLong[0][fpga],FsopLong[1][fpga],FsopShort[fpga]);
	  bool APV1Bad = (APV1WrongAdd );
	  bool APV2Bad = (APV2WrongAdd );
	  bool fiberBad = (outOfSync || unlocked);
          bool anyError = APV1Error || APV1WrongAdd || APV2Error || APV2WrongAdd || fiberBad;
	  if (anyError) {
	    problemsSeen++;
	    feErr++;
	    errors[*ifed][fpga]->Fill(fi);
	    std::cout << "EVT#" << iEvent.id().event() << " FED#" << *ifed << " FE#" << fpga << " Fiber#" << fi << " Status Word: ";
	    std::cout << "\tAPV1 Error? " << (APV1Error?"Yes":"No");
	    std::cout << "\tAPV1 Bad Address? " << (APV1WrongAdd?"Yes":"No");
	    std::cout << "\tAPV2 Error? " << (APV2Error?"Yes":"No");
	    std::cout << "\tAPV2 Bad Address? " << (APV2WrongAdd?"Yes":"No");
	    std::cout << "\tOut Of Sync? " << (outOfSync?"Yes":"No");
	    std::cout << "\tFiber Not Locked?\t" << (unlocked?"Yes":"No");
	    std::cout << std::endl;
	  }
          if ( (!fiberBad) && (APV1Bad || APV2Bad) ) {
            FiberWHApv[*ifed][fpga]->Fill(fi);
          }
	  nFi++;
	  if ((!unlocked) && (!outOfSync)) {
	    goodFibers++;
	    if ( !APV1WrongAdd) goodAPVs++;
	    if ( !APV2WrongAdd) goodAPVs++;
	  }
	}//fiber loop----------------------------------------------------------------------------------------------

	//counter for the number of wrong headers
	WHError[*ifed][fpga] = feErr;	

	//FE Address verification portion--------------------------------------------------------------------------- 
	//fill the matrix of FE addresses for all FE FPGAs ---------------------------------------------------------
	feMajorAddress[*ifed][fpga] = (FsopShort[fpga]>>8);		

	//		std::cout<<" VECADDS1 " <<" fed no: "<<*ifed<<" "<<feMajorAddress[*ifed][fpga]<<std::endl;
	//----------------------------------------------------------------------------------------------------------
	//all fes on one plot as in the plot # one from Mersi-------------------------------------------------------
	//set the bin contents every N events(all for now)----------------------------------------------------------
	AddCheck0->setBinContent( feCtr, (FsopShort[fpga]>>8) ); 
	
	feMedianAddr[feCtr]= (FsopShort[fpga]>>8);
	//----------------------------------------------------------------------------------------------------------

	for(int i = 31; i >= 0; i--){
	  //block describing the 0 - 31 bits of what we have going on
	  // (for humans Fill( n +1 ) where (n+1) = Fiber Number------	
	  if( ( 0x1 & (FsopLong[1][fpga] >> i) ) == 0 ){
				
	    if ( !( i%6 ) ){//APV Error B - APV 0
	      FiberStatusBits[fpga][0][*ifed]->Fill( i/6 );
	    }
	    if( !( (i-1)%6 ) ){//Wrong Header B APV 0
	      FiberStatusBits[fpga][1][*ifed]->Fill( ((i-1)/6) );
	      badApvCounter++;
	      rfeErr++;
	    }
	    if( !( (i-2)%6 ) && i <= 26 ){//APV Error B APV 1
	      FiberStatusBits[fpga][2][*ifed]->Fill( ((i-2)/6) );
	    }
	    if( !( (i-3)%6 ) && i <= 27 ){//Wrong Header B APV 1
	      FiberStatusBits[fpga][3][*ifed]->Fill( ((i-3)/6) );
	      badApvCounter++;
	      rfeErr++;
	    } 
	    if( !( (i-4)%6 ) && i <= 28 ){//Out Of Synch B
	      FiberStatusBits[fpga][4][*ifed]->Fill( ((i-4)/6) );
	      goodApvCounter += 2;
	      oos += 2;
	      syErr++;
	      //oosFedEvent->Fill(fedCounter,iEvent.id().event()); 
	      //oosFedEvent->Fill(iEvent.id().event(),fedCounter); 
	    }
	    if( !( (i-5)%6 ) && i <= 29){//Lock
	      FiberStatusBits[fpga][5][*ifed]->Fill( ((i-5)/6) );
	      nolock +=2;
	      lkErr++;
	    }
	  }	
	  //--------------------------------------------------------------------------------------------------------
	  //block describing the 32 - 64 bits of what we have going on----------------------------------------------	
	  if( ( 0x1 & (FsopLong[0][fpga] >> i) ) == 0 ){
	    if ( !( (i-4)%6 ) && i<=28 ){//APV Error B - APV 0
	      FiberStatusBits[fpga][0][*ifed]->Fill( 6 + ((i-4)/6) );
	    }
	    if( !( (i-5)%6 ) && i<= 29){//Wrong Header B APV 0
	      FiberStatusBits[fpga][1][*ifed]->Fill( 6 + ((i-5)/6) );
	      badApvCounter++;
	      rfeErr++;
	    }
	    if( !( i%6 ) ){//APV Error B APV 1
	      FiberStatusBits[fpga][2][*ifed]->Fill( 5 + (i/6) );
	    }
	    if( ! ((i-1)%6) ){//Wrong Header B APV1
	      FiberStatusBits[fpga][3][*ifed]->Fill( 5 + ((i-1)/6) );
	      badApvCounter++;
	      rfeErr++;
	    } 
	    if( !( (i-2)%6 ) && i <= 26 ){//Out of Synch
	      FiberStatusBits[fpga][4][*ifed]->Fill( 5 + ((i-2)/6) );
	      goodApvCounter += 2;
	      oos += 2;
	      syErr++;
	      //oosFedEvent->Fill(fedCounter,iEvent.id().event()); 
	      //oosFedEvent->Fill(iEvent.id().event(),fedCounter); 
	    }
	    if( !( (i-3)%6 ) && i <= 27 ){//Lock
	      FiberStatusBits[fpga][5][*ifed]->Fill( 5 + ((i-3)/6) );
	      nolock +=2;
	      lkErr++;
	    }
	  }	
	  //-------------------------------------------------------------------------------------------------------
	}
	
	for(int i = 7; i >= 0; i--){
	  if( ( 0x1 & (FsopShort[fpga] >> i) ) == 0 ){
	    if( i == 2 ){//APV Error B APV 0
	      FiberStatusBits[fpga][0][*ifed]->Fill( 11 );
	    }
	    if( i == 3 ){//Wrong Header B APV 0
	      FiberStatusBits[fpga][1][*ifed]->Fill( 11 );
	      badApvCounter++;
	      rfeErr++;
	    }
	    if( i == 4 ){//APV Error B APV 1
	      FiberStatusBits[fpga][2][*ifed]->Fill( 11 );
	    }
	    if( i == 5 ){//Wrong Header B APV 1
	      FiberStatusBits[fpga][3][*ifed]->Fill( 11 );
	      badApvCounter++;
	      rfeErr++;
	    }
	    if( i == 6 ){//Out of Synch
	      FiberStatusBits[fpga][4][*ifed]->Fill( 11 );
	      goodApvCounter += 2;
	      oos += 2;
	      syErr++;
	      //oosFedEvent->Fill(iEvent.id().event(),fedCounter); 
	    }
	    if( i == 7 ){//Lock
	      FiberStatusBits[fpga][5][*ifed]->Fill( 11 );
	      nolock +=2;
	      lkErr++;
	    }
	    if( i == 0 ){//Out of Synch Fib 11
	      FiberStatusBits[fpga][4][*ifed]->Fill( 10 );
	      goodApvCounter += 2;
	      oos += 2;
	      syErr++;
	      //oosFedEvent->Fill(iEvent.id().event(),fedCounter); 
	    }
	    if( i == 1 ){//Lock Fib 11
	      FiberStatusBits[fpga][5][*ifed]->Fill( 10 );
	      nolock +=2;
	      lkErr++;
	    }
	
	  }
	}
	
	//matrices for raw header, synch and lock errors

	LKError[*ifed][fpga] = lkErr;	
	SYError[*ifed][fpga] = syErr;	
	RWHError[*ifed][fpga] = rfeErr;	


      }//if FE conditonal loop
    }//for FE fpga loop 
		
    //XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    //for the fed id # , plot the # of oos instances

    OosPerFed[*ifed]->Fill( iEvent.id().event(), oos   );
	
		
    //reporting the majority APV address of the FED

    std::cout<<dec<<" FED NUMBER : "<<*ifed<<" CTR "<<feCtr<<" # FE enabled : "<<(feCtr - tmpCtr)<<" FEDKTR "<<fedCounter<<std::endl;
    std::cout<<dec<<" CTR: "<<feCtr<<std::endl;
    std::cout<<dec<<"For EVT "<<iEvent.id().event()<<std::endl;
    //	std::cout<<dec<<" Mapping "<<*ifed<<" to Bin No "<< fedCtr <<std::endl;
    tmpCtr = feCtr;
    std::cout<<std::endl;

    //XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    //--------------------------------------------------------------------------------------------------------

    //"clears" the stream FED ID stream that is....
    ss.str(" ");
    //reset fed counter
    //	fedCtr = 0;

    //end of DQM plotting portion+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  } // end of the for "ifed" 4 loop***************************************************************************
  ///Nicks histo good apv action------------------------------------------------------------------------------
  int nAPVs = goodFibers*2;
  double goodAPVper = (double(nAPVs - goodAPVs)) / double(nAPVs) * double(100);
  //std::cout << "Event: " << iEvent.id().event() << " Percentage of APVs which are on good fibers but have bad addresses: " << goodAPVper << std::endl;
  goodAPVsPerEvent_->setBinContent(iEvent.id().event(),double(100)-goodAPVper);
  std::cout << "Total enabled channels = " << nFi << '\t' << "Total good channels = " << goodFibers << std::endl;
  APVProblemCounter_ += (nAPVs - goodAPVs);


  std::cout<<"GPRNUM"<<(double(100)-goodAPVper)<<" for event "<<iEvent.id().event()<<std::endl;


  apvPrct.push_back(  double(100)-goodAPVper   ); // for the end job mode , etc.

  //Per Event-------------------------------------------------------------------------------------------------

  std::cout << "Num of APVs: " << 2*total_enabled_channels <<std::endl;

  //Debug Stuff.......

  feMedianAddr.resize(feCtr); // resizes the vector to the number of FE FPGAs enabled

  sort(feMedianAddr.begin(), feMedianAddr.end());
	
  std::cout << "MIDLOC" << (feCtr / 2) << " " << " feCtr: " << feCtr << " vector.size "
       << feMedianAddr.size() <<" evtnumber : " << iEvent.id().event() << std::endl;	
	
  std::cout<<"ENUM"<<iEvent.id().event()<<std::endl;
  medianAddr = feMedianAddr[(feCtr / 2)];
	
  std::cout<<"MEDADDR"<<medianAddr<<std::endl;	
  /*
    for(int i = 0; i < feCtr; i++){
    std::cout<<"VK# "<<" "<<i<<" "<<feMedianAddr[i]<<std::endl;
    }
  */
  for(int i = 0; i < feCtr; i++){
    if(feMedianAddr[i] == medianAddr){
      goodFe++;
    }
  }
  std::cout<<" GF "<<goodFe<<std::endl;
  std::cout<<" COU "<<feCtr<<std::endl;

  if(feCtr != 0 ){
    prct = ( double(goodFe) / double(feCtr) );
    std::cout<<" PRCT "<< prct <<" for : "<<iEvent.id().event()<<std::endl;
  }
  //Sets up the % FEs ok per event plot

  if(prct){

    if(iEvent.id().event() < 1001){
      AddConstPerEvent->setBinContent(iEvent.id().event() , prct );
    }
  }
  //Plot # 3 Portion

  int tempn; //variable sotrage for fed number from vector
  int tempfen; //variable sotrage for feEnabled number from vector

  //More Debug Stuff-------------------------------------------------------------------------------------------
  /*
    for(int i = 0; i < fenumbers.size(); i++){
    tempn = fenumbers[i];
    std::cout<<" NUMBERING " <<tempn<<std::endl;
    }
    for(int i = 0; i < fenumbers.size(); i++){
    tempfen = fen[i];
    std::cout<<" FENUMBERING " <<tempfen<<std::endl;
    }
	

    vector<uint16_t>::const_iterator ifed2 = cabling->feds().begin();

    std::cout<<" PRELIM1 "<<std::endl;

    for ( ; ifed2 != cabling->feds().end(); ifed2++ ) {

    std::cout<<" PRELIM2 "<<std::endl;

    std::cout<<"LOWFEEN"<<feEnable<<std::endl;
  */

  //out of synch 2 d plot





  //apparently working version of the median address error plots analogy to APVE plots---------------------------




  stringstream fss; 

  for(int i = 0; i < fednumbers.size(); i++){
	
    fss <<	fednumbers[i];

    tempn = fednumbers[i];
		
    tempfen = fen[i]; //front end enabled

    CumNumber->setBinLabel(i+1, fss.str());
    CumNumber1->setBinLabel(i+1, fss.str());
    CumNumber2->setBinLabel(i+1, fss.str());
    CumNumber3->setBinLabel(i+1, fss.str());
    CumNumber4->setBinLabel(i+1, fss.str());



    for(int fpga = 0; fpga < 8 ; fpga++){

      if( 0x1 & (tempfen >> fpga) ){
	std::cout<<" VECADDS2 " <<" fed no: "<<tempn<<" "<<feMajorAddress[tempn][fpga]<<std::endl;
	std::cout<<" WHERRS "<<" fed no: "<<tempn<<" fpga no :"<<fpga<<WHError[tempn][fpga]<<std::endl;
	std::cout<<" LKERRS "<<" fed no: "<<tempn<<" fpga no :"<<fpga<<LKError[tempn][fpga]<<std::endl;
	std::cout<<" SYERRS "<<" fed no: "<<tempn<<" fpga no :"<<fpga<<SYError[tempn][fpga]<<std::endl;
	std::cout<<" RWHERRS "<<" fed no: "<<tempn<<" fpga no :"<<fpga<<RWHError[tempn][fpga]<<std::endl;
	if( WHError[tempn][fpga] != 0){
	  CumNumber1->Fill(i);	                
	  FeWHApv[tempn]->Fill(fpga);
	}					 
	if( LKError[tempn][fpga] != 0){
	  CumNumber2->Fill(i);	                
	  FeLKErr[tempn]->Fill(fpga);
	}					 
	if( SYError[tempn][fpga] != 0){
	  CumNumber3->Fill(i);	                
	  FeSYErr[tempn]->Fill(fpga);
	}					 
	if( RWHError[tempn][fpga] != 0){
	  CumNumber4->Fill(i);	                
	  FeRWHErr[tempn]->Fill(fpga);
	}					 
					
	if( feMajorAddress[tempn][fpga] != medianAddr  ){
	  std::cout<<" FILLED "<<tempn<<std::endl;
	  CumNumber->Fill(i); 
	  FeMajApvErr[tempn]->Fill(fpga) ;
	}
   
			
      }
    }


	

    fss.str(" ");


  }//fed numbering loop

  //APVs in synch PRCTS + Low and High and MODE Calc-----------------------------------------------------------------

  double apvperct = 0;
  double apvperct1 = 0;
  double apvperct2 = 0;
  int normalize = 0;
  int normalize2 = 0;

  normalize = ( badApvCounter - goodApvCounter);//basically out of synch bit set 
  normalize2 = ( badApvCounter -  nolock); 

  std::cout<<"NRML"<<normalize<<" for event # "<<iEvent.id().event()<<std::endl;
  std::cout<<"NRML2"<<normalize2<<" for event # "<<iEvent.id().event()<<std::endl;
  std::cout<<"BIZZY"<<badApvCounter<<" for event # "<<iEvent.id().event()<<std::endl;
  std::cout<<"GIZZY"<<goodApvCounter<<" for event # "<<iEvent.id().event()<<std::endl;

	
  apvperct = 100.0*( ( double(2*total_enabled_channels) - double( badApvCounter ) ) / double(2*total_enabled_channels) ); 	
  apvperct2 = 100.0*( ( double(2*total_enabled_channels) - double( normalize2 ) ) / double(2*total_enabled_channels) ); 	
  apvperct1 = 100.0*( ( double(2*total_enabled_channels) - double( normalize ) ) / double(2*total_enabled_channels) ); 	
	
  std::cout<<" APVPRCT "<<apvperct<<" EvT "<<iEvent.id().event()<<std::endl;
  std::cout<<" NOLOCK "<<nolock<<" EvT "<<iEvent.id().event()<<std::endl;
  std::cout<<" BADAPVCOUNTER "<<badApvCounter<<" EvT "<<iEvent.id().event()<<std::endl;

  //	apvPrct.push_back(goodAPVper); // for the end job mode , etc.
	
	
	

  if(iEvent.id().event() < 1001){
    ApvAddConstPerEvent->setBinContent(iEvent.id().event() , apvperct  );
    ApvAddConstPerEvent1->setBinContent(iEvent.id().event() , apvperct1  );
    ApvAddConstPerEvent2->setBinContent(iEvent.id().event() , apvperct2  );
    NoLock->setBinContent(iEvent.id().event() , nolock  );
    BadHead->setBinContent(iEvent.id().event() , badApvCounter  );
    NoSynch->setBinContent(iEvent.id().event() , oos  );
  }

  //-------------------------------------------------------------------------------------------------------------
  //feMedianAddr.clear(); does not seem to be necessary
  feCtr = 0; // reset this puppy to 0 for good measure
  tmpCtr = 0;
  goodFe = 0;
  badApvCounter = 0;
  nolock = 0;
  goodApvCounter = 0;
  oos = 0;
}//End of the Event Loop ("analyze function" Called once per event


// ------------ method called once each job just before starting event loop  ------------
void 
CnBAnalyzer::beginJob(const edm::EventSetup& iSetup)
{

  if ( useCabling_ ) {

    // Retrieve FED cabling
    vector<uint16_t> fed_ids;

    //Retrieve FED ids from cabling map and iterate through 
    edm::ESHandle<SiStripFedCabling> cabling;
    iSetup.get<SiStripFedCablingRcd>().get( cabling );

    vector<uint16_t>::const_iterator ifed = cabling->feds().begin();
    for ( ; ifed != cabling->feds().end(); ifed++ ) { fed_ids.push_back( *ifed ); }

    histoNaming( fed_ids, 23 ); // default value set for now

  } 
  
}

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
    //-------------------------------------------------------------------------------------------------------------------------
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
		

}// 

// ------------ method called once each job just after ending the event loop  ------------
void 
CnBAnalyzer::endJob() {

  for (unsigned int i = 0; i < ApveErrCount.size(); i++ ) {
    delete ApveErrCount[i];
  }

  dbe->showDirStructure();
  dbe->save(fileName_);
	

  int sz = apvPrct.size();

  std::cout<<" SIZZY "<<sz<<std::endl;

  sort(apvPrct.begin(), apvPrct.end());
  std::cout<<" MIZZY "<<apvPrct[sz - 1]<<std::endl;
  std::cout<<" LIZZY "<<apvPrct[0]<<std::endl;

  map<double,int> m1;
  for(pi = apvPrct.begin(); pi != apvPrct.end(); pi++)
    {
      m1[*pi]++;
    }
  int prevHighCount = 0;
  float answer =0 ;
  for(std::map<double,int>::iterator i = m1.begin(); i != m1.end(); i++){
    if((*i).second > prevHighCount){
      prevHighCount = (*i).second;
      answer = (*i).first;
    }
  }



  std::cout << " MOOZY " << answer << std::endl;



}

bool
CnBAnalyzer::getBit(int bitNumber, Fed9U::u32 FsopLongHi, Fed9U::u32 FsopLongLow, Fed9U::u16 FsopShort) {
  unsigned char result = 0;
  if (bitNumber<32)
    result = (FsopLongLow >> bitNumber) & 0x1;
  if ( bitNumber>=32 && bitNumber<64 )
    result = (FsopLongHi >> (bitNumber-32)) & 0x1;
  if ( bitNumber>=64 && bitNumber <80)
    result = (FsopShort >> (bitNumber-64)) & 0x1;
  return (result != 0x0);
}


