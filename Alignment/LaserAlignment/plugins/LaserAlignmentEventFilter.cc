#include "LaserAlignmentEventFilter.h"

#include <FWCore/Framework/interface/Event.h> 
#include <FWCore/Framework/interface/EventSetup.h> 
#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/EventSetupRecordKey.h>
#include <FWCore/Framework/interface/ESHandle.h>

#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBuffer.h"

#include <algorithm>

///
/// constructors and destructor
///
LaserAlignmentEventFilter::LaserAlignmentEventFilter( const edm::ParameterSet& iConfig ) :
  FED_collection(iConfig.getParameter<edm::InputTag>("FedInputTag")),
  signal_filter(true),
  single_channel_thresh(11),
  channel_count_thresh(4),
  LAS_event_count(0),
  cabling(0),
  cacheId_(0)
{

  FED_collection_token = consumes<FEDRawDataCollection>(FED_collection);

  // Read in Filter Lists
  std::vector<int> FED_IDs = iConfig.getParameter<std::vector<int> >("FED_IDs");
  set_las_fed_ids(FED_IDs);

  std::vector<int> SIGNAL_IDs = iConfig.getParameter<std::vector<int> >("SIGNAL_IDs");
  set_las_signal_ids(SIGNAL_IDs);

  // Read in Filter Flags
  signal_filter = iConfig.getParameter<bool>("SIGNAL_Filter");

  // Read in Filter Parameters
  single_channel_thresh =  iConfig.getParameter<unsigned>("SINGLE_CHANNEL_THRESH");
  channel_count_thresh = iConfig.getParameter<unsigned>("CHANNEL_COUNT_THRESH");

  edm::LogInfo("LasFilterConstructor") << "\n" 
				       << "\nSIGNAL_Filter: " << (signal_filter ? "true" : "false")
				       << ", SIGNLE_CHANNEL_THRESH: " << single_channel_thresh 
				       << ", CHANNEL_COUNT_THRESH: " << channel_count_thresh;
}

///
///
///
LaserAlignmentEventFilter::~LaserAlignmentEventFilter() {
}

///
///
///
void LaserAlignmentEventFilter::beginRun( const edm::EventSetup& iSetup) {
  updateCabling( iSetup );
}

///
/// Checks for Laser Signals in specific modules
/// For Accessing FED Data see also EventFilter/SiStripRawToDigi/src/SiStripRawToDigiUnpacker.cc and related files

bool LaserAlignmentEventFilter::filter( edm::Event& iEvent, const edm::EventSetup& iSetup ) 
{
  updateCabling( iSetup );
  unsigned int det_ctr=0; // Count how many modules are tested for signal
  unsigned int sig_ctr=0; // Count how many modules have signal
  unsigned long buffer_sum=0; // Sum of buffer sizes

  // Retrieve FED raw data (by label, which is "source" by default)
  edm::Handle<FEDRawDataCollection> buffers;
  iEvent.getByToken( FED_collection_token, buffers ); 


  std::vector<uint16_t>::const_iterator ifed = las_fed_ids.begin();
  for ( ; ifed != las_fed_ids.end(); ifed++ ) {
    // Retrieve FED raw data for given FED 
    const FEDRawData& input = buffers->FEDData( static_cast<int>(*ifed) );
    LogDebug("LaserAlignmentEventFilter") << "Examining FED " << *ifed; 

     // Check on FEDRawData pointer
     if ( !input.data() ) {
       continue;
     }	
    
     // Check on FEDRawData size
     if ( !input.size() ) {
       continue;
     }

     // get the cabling connections for this FED
     auto conns = cabling->fedConnections(*ifed);

     // construct FEDBuffer
     std::auto_ptr<sistrip::FEDBuffer> buffer;
     try {
       buffer.reset(new sistrip::FEDBuffer(input.data(),input.size()));
       if (!buffer->doChecks()) {
 	throw cms::Exception("FEDBuffer") << "FED Buffer check fails for FED ID " << *ifed << ". (BW)";
	if(buffer->daqEventType() != sistrip::DAQ_EVENT_TYPE_CALIBRATION){
	  edm::LogWarning("LaserAlignmentEventFilter") << "Event is not calibration type";
	  return false;
	}
       }
     }
     catch (const cms::Exception& e) { 
       if ( edm::isDebugEnabled() ) {
 	edm::LogWarning("LaserAlignmentEventFilter") << "Exception caught when creating FEDBuffer object for FED " << *ifed << ": " << e.what();
       }
       continue;
     }

    // Iterate through FED channels, extract payload and create Digis
    std::vector<FedChannelConnection>::const_iterator iconn = conns.begin();
    for ( ; iconn != conns.end(); iconn++ ) {

      if(signal_filter){
	if ( std::binary_search(las_signal_ids.begin(), las_signal_ids.end(), iconn->detId())){ 
	  LogDebug("LaserAlignmentEventFilter")
	    << " Found LAS signal module in FED " 
	    << *ifed
	    << "  DetId: " 
	    << iconn->detId() << "\n"
	    << "buffer->channel(iconn->fedCh()).size(): " << buffer->channel(iconn->fedCh()).length();
	  buffer_size.push_back(buffer->channel(iconn->fedCh()).length());
	  det_ctr ++;
	  if(buffer->channel(iconn->fedCh()).length() > single_channel_thresh) sig_ctr++;
	  buffer_sum += buffer->channel(iconn->fedCh()).length();
	  if(sig_ctr > channel_count_thresh){
	    LAS_event_count ++;
	    LogDebug("LaserAlignmentEventFilter") << "Event identified as LAS";
	    return true;
	  }
	}
      }
    } // channel loop
  } // FED loop
  
//   LogDebug("LaserAlignmentEventFilter") << det_ctr << " channels were tested for signal\n" 
// 			    <<sig_ctr << " channels have signal\n"
// 			    << "Sum of buffer sizes: " << buffer_sum;


  return false;
}


///
///
///
void LaserAlignmentEventFilter::endJob() {
  //edm::LogInfo("LaserAlignmentEventFilter") << "found " << LAS_event_count << " LAS events";
}

// Create the table of FEDs that contain LAS Modules
void LaserAlignmentEventFilter::set_las_fed_ids(const std::vector<int>& las_feds)
{
  // Convert the std::vector to a std::set
  las_fed_ids = std::vector<uint16_t>(las_feds.begin(), las_feds.end());
  // Sort for binary search
  std::sort(las_fed_ids.begin(), las_fed_ids.end());
}

// Create table of modules to be tested for signals
void LaserAlignmentEventFilter::set_las_signal_ids(const std::vector<int>& las_signal)
{
  // Convert the std::vector to a std::set
  las_signal_ids = std::vector<uint32_t>(las_signal.begin(), las_signal.end());
  // Sort for binary search
  std::sort(las_signal_ids.begin(), las_signal_ids.end());
}


// This method was copied from EventFilter/SiStripRawToDigi/plugin/SiStripRawToDigiModule
void LaserAlignmentEventFilter::updateCabling( const edm::EventSetup& setup ) {

  uint32_t cache_id = setup.get<SiStripFedCablingRcd>().cacheIdentifier();

  if ( cacheId_ != cache_id ) {
   edm::ESHandle<SiStripFedCabling> c;
   setup.get<SiStripFedCablingRcd>().get( c );
   cabling = c.product();
   cacheId_ = cache_id;
  }
}


//define this as a plug-in
DEFINE_FWK_MODULE(LaserAlignmentEventFilter);
