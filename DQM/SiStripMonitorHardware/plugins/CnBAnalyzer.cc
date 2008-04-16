#include "DQM/SiStripMonitorHardware/plugins/CnBAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DQM/SiStripMonitorHardware/interface/Fed9UEventAnalyzer.hh"
#include "DQM/SiStripMonitorHardware/interface/Fed9UDebugEvent.hh"

#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h" 

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"

#include "DQMServices/Core/interface/DQMStore.h"

// This is the maximum number of histogrammed FEDs
// If the number of FEDs exceeds this limit we have a crash
//#define N_MAX_FEDS  (1024)
//#define N_MAX_FEDUS (N_MAX_FEDS * 8)
// feMajorAddress( N_MAX_FEDS,vector<uint16_t>(8) ), // a grand total of ~ 4000 front end units
// feMedianAddr(N_MAX_FEDUS),

CnBAnalyzer::CnBAnalyzer(const edm::ParameterSet& iConfig) {
  // Get hold of back-end interface
  // dbe = edm::Service<DaqMonitorBEInterface>().operator->();

  // Dqm private object
  dqm_ = NULL;
  
  // Parameters for working with S-link and dumping the hex buffer
  swapOn_ = iConfig.getUntrackedParameter<int>("swapOn");
  outputFileName_ = iConfig.getUntrackedParameter<string>("rootFile", "");
  outputFileDir_ = iConfig.getUntrackedParameter<string>("rootFileDirectory","");
  
  // FED address mapping is obtained through
  // FEDNumberting object: use and throw !
  FEDNumbering fedNum;
  // valid FedIds for the tracker
  fedIdBoundaries_ = fedNum.getSiStripFEDIds();
  totalNumberOfFeds_ = fedIdBoundaries_.second - fedIdBoundaries_.first + 1;
}

CnBAnalyzer::~CnBAnalyzer() {
  // Go to top directory
  dqm()->cd();
  // and remove MEs at top directory
  dqm()->removeContents(); 
}

void CnBAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){

  using namespace edm;
  using namespace std;

  // TODO: refuse to run if we are in Scope mode

  // Retrieve FED raw data ("source" label is now fixed by framework)
  edm::Handle<FEDRawDataCollection> buffers;
  iEvent.getByType( buffers );

  fedIds_.clear();
  for (uint16_t ifed = fedIdBoundaries_.first ; ifed <= fedIdBoundaries_.second; ifed++ ) {
    if ( buffers->FEDData( static_cast<int>(ifed) ).size() >= Fed9U::Fed9UDebugEvent::MinimumBufferSize ) {
      fedIds_.push_back(ifed);
    }
  }
  
#ifdef CNBANALYZER_DEBUG
  LogInfo("FEDBuffer") << "Number of Tracker Fed Buffers:" << fedIds_.size();
  LogInfo("FEDBuffer") << "EVENTNUMB: " << iEvent.id().event();
#endif


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
    
#ifdef CNBANALYZER_DEBUG
    LogInfo("FEDBuffer") << "A FED event: ";
#endif
    createDetailedFedHistograms((*ifed));
    
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
      
      LogInfo("FEDBuffer") << "FEDevent correctly initialized";
#ifdef CNBANALYZER_DEBUG
#endif
      
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
  
#ifdef CNBANALYZER_DEBUG
      LogInfo("FEDBuffer") << "total_enabled = " << total_enabled_channels;
      LogInfo("FEDBuffer") << "total_faulty = " << total_faulty_channels;
      LogInfo("FEDBuffer") << "internalFreeze = " << thisFedEventErrs.internalFreeze;
      LogInfo("FEDBuffer") << "bxError = " << thisFedEventErrs.bxError;
#endif
      
      // Fill the Front-end failure counters 
      for (unsigned int iFrontEnd=0; iFrontEnd<8; iFrontEnd++) {
#ifdef CNBANALYZER_DEBUG
	LogInfo("FEDBuffer") << "feOverflow[" << iFrontEnd << "] = " << thisFedEventErrs.feOverflow[iFrontEnd];
	LogInfo("FEDBuffer") << "apvAddressError[" << iFrontEnd << "] = " << thisFedEventErrs.apvAddressError[iFrontEnd];
#endif

	if (thisFedEventErrs.feOverflow[iFrontEnd])
	  feOverFlow_[*ifed]->Fill(iFrontEnd*24+1);
	if (thisFedEventErrs.apvAddressError[iFrontEnd])
	  feAPVAddr_[*ifed]->Fill(iFrontEnd*24+1);
	
      }
      
      // Fill the channel failure counters 
      for (unsigned int iChannel=0; iChannel<96; iChannel++) {

#ifdef CNBANALYZER_DEBUG
	LogInfo("FEDBuffer") << "channel[" << iChannel << "] = " << hex << thisFedEventErrs.channel[iChannel];
#endif

	if (thisFedEventErrs.channel[iChannel]==Fed9UEventAnalyzer::FIBERUNLOCKED)
	  chanErrUnlock_[*ifed]->Fill(iChannel*2+1);
	if (thisFedEventErrs.channel[iChannel]==Fed9UEventAnalyzer::FIBEROUTOFSYNCH)
	  chanErrOOS_[*ifed]->Fill(iChannel*2+1);	
      }      
      
    
      // apv[96*2]
      for (unsigned int iApv=0; iApv<192; iApv++) {
#ifdef CNBANALYZER_DEBUG
	LogInfo("FEDBuffer") << "apv[" << iApv << "] = " << hex << thisFedEventErrs.apv[iApv];
#endif

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

} // End of the Event Loop (analyze: called once per event)


// ------------ method called once each job just before starting event loop  ------------
void 
CnBAnalyzer::beginJob(const edm::EventSetup& iSetup)
{
  
  createRootFedHistograms();

}

// The following method should be called
// at the job initialization by beginJob()
void CnBAnalyzer::createRootFedHistograms() {

  std::string baseFolder;
  baseFolder = sistrip::root_ + "/Fed Monitoring Summary";

  dqm()->setCurrentFolder(baseFolder);

  // This Histogram will be filled with 
  // problemsSeen / totalChannels
  fedGenericErrors_ = dqm()->book1D( "Fed Generic Errors","Fed Generic Errors vs. FED #",
				     totalNumberOfFeds_,
				     fedIdBoundaries_.first  - 0.5,
				     fedIdBoundaries_.second + 0.5 );
  
  
  // bool internalFreeze
  fedFreeze_ = dqm()->book1D( "Fed Freeze","Fed Freeze vs. FED #",
			      totalNumberOfFeds_,
			      fedIdBoundaries_.first  - 0.5,
			      fedIdBoundaries_.second + 0.5 );
  
  // bool bxError
  fedBx_ = dqm()->book1D( "Fed Bx Error","Fed Bx Error vs. FED #",
			  totalNumberOfFeds_,
			  fedIdBoundaries_.first  - 0.5,
			  fedIdBoundaries_.second + 0.5 );

  // Trend plots:
  totalChannels_  = dqm()->book1D( "Total channels vs. Event",
				   "Total channels vs. Event for all FEDs",
				   1001, 0.5, 1000.5);

  faultyChannels_ = dqm()->book1D( "Faulty channels vs. Event",
				   "Faulty channels vs. Event for all FEDs",
				   1001, 0.5, 1000.5);

}

void CnBAnalyzer::createDetailedFedHistograms( const uint16_t& fed_id ) {

  std::map<uint16_t, bool>::iterator itFeds;

  itFeds=foundFeds_.find(fed_id);

  if (itFeds==foundFeds_.end()) {

    SiStripFedKey thisFedKey(fed_id, 0, 0, 0);

    foundFeds_[fed_id]=true;
    
    stringstream fedNumber;
    fedNumber << fed_id;
    
    // Set working directory prior to booking histograms 
    std::string dir = thisFedKey.path();
    dqm()->setCurrentFolder( dir );
    
    // All the following histograms are such that thay can be plot together
    // In fact the boundaries of the plots are 1, 192 (or actually 0.5, 192,5)
    // They have a different binning, though, which reflect the granularity of the system.
    
    // When filling these plots one can access directly the bin (for example feOverFlow, bin number 8
    // to address the front-end unit number 8) or one can make use of the x axis, which correspond
    // to the APV index (1, 192).
    
    
    //   bool feOverflow[8];
    feOverFlow_[fed_id]     = dqm()->book1D( "FedUnit Overflow",
					     "FedUnit Overflow for FED #"+fedNumber.str(),
					     8, 0.5, 192.5 );
    
    //   bool apvAddressError[8];
    feAPVAddr_[fed_id]      = dqm()->book1D( "APV Address error",
					     "FedUnit APV Address error for FED #"+fedNumber.str(),
					     8, 0.5, 192.5 );
    
    // Channel[96]
    chanErrUnlock_[fed_id]  = dqm()->book1D( "Unlock error",
					     "Unlocked Fiber error for FED #"+fedNumber.str(),
					     96, 0.5, 192.5);
    chanErrOOS_[fed_id]     = dqm()->book1D( "OOS error",
					     "OutOfSynch Fiber error for FED #"+fedNumber.str(),
					     96, 0.5, 192.5);
    
    // apv[96*2]
    badApv_[fed_id]         = dqm()->book1D( "Bad APV error",
					     "Bad APV error for FED #"+fedNumber.str(),
					     192, 0.5, 192.5);


  } // Otherwise we have nothing to do
  
}


// ------------ method called once each job just after ending the event loop  ------------
void CnBAnalyzer::endJob() {

  if (outputFileName_!="") {
    dqm()->showDirStructure();
    std::string completeFileName = outputFileDir_ + std::string("/test_") + outputFileName_;
    dqm()->save(completeFileName);
  }

}


DQMStore* const CnBAnalyzer::dqm( std::string method ) const {
  if ( !dqm_ ) { 
    std::stringstream ss;
    if ( method != "" ) { ss << "[CnBAnalyzer::" << method << "]" << std::endl; }
    else { ss << "[CnBAnalyzer]" << std::endl; }
    ss << " NULL pointer to DQMStore";
    edm::LogWarning("SiStripMonitorHardware") << ss.str();
    return 0;
  } else { return dqm_; }
}
