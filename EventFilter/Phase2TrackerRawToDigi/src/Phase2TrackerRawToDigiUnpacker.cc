#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerRawToDigiUnpacker.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/src/fed_header.h"
#include "DataFormats/FEDRawData/src/fed_trailer.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripEventSummary.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
// #include "EventFilter/Phase2TrackerRawToDigi/interface/TFHeaderDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <ext/algorithm>

// To use old configuration uncoment CBC_RAWTODIGI definition 
#define CBC_RAWTODIGI

namespace sistrip {

  RawToDigiUnpacker::RawToDigiUnpacker( int16_t appended_bytes, int16_t fed_buffer_dump_freq, int16_t fed_event_dump_freq, int16_t trigger_fed_id,
                                        bool using_fed_key, bool unpack_bad_channels, bool mark_missing_feds, const uint32_t errorThreshold ) :
    headerBytes_( appended_bytes ),
    fedBufferDumpFreq_( fed_buffer_dump_freq ),
    fedEventDumpFreq_( fed_event_dump_freq ),
    triggerFedId_( trigger_fed_id ),
    useFedKey_( using_fed_key ),
    unpackBadChannels_( unpack_bad_channels ),
    markMissingFeds_( mark_missing_feds ),
    event_(0),
    once_(true),
    first_(true),
    useDaqRegister_(false),
    quiet_(true),
    extractCm_(false),
    doFullCorruptBufferChecks_(false),
    doAPVEmulatorCheck_(true),
    errorThreshold_(errorThreshold)
  {
    if ( edm::isDebugEnabled() ) {
      LogTrace("Phase2TrackerRawToDigi")
	<< "[sistrip::RawToDigiUnpacker::"<<__func__<<"]"
	<<" Constructing object...";
    }
    if (unpackBadChannels_) {
      edm::LogWarning("Phase2TrackerRawToDigi") << "Warning: Unpacking of bad channels enabled. Only enable this if you know what you are doing. " << std::endl;
    }
  }

  RawToDigiUnpacker::~RawToDigiUnpacker() {
    if ( edm::isDebugEnabled() ) {
      LogTrace("Phase2TrackerRawToDigi")
	<< "[sistrip::RawToDigiUnpacker::"<<__func__<<"]"
	<< " Destructing object...";
    }
  }

  void RawToDigiUnpacker::createDigis( const SiStripFedCabling& cabling, const FEDRawDataCollection& buffers, SiStripEventSummary& summary, RawDigis& scope_mode, RawDigis& virgin_raw, RawDigis& proc_raw, Digis& zero_suppr, DetIdCollection& detids, RawDigis& cm_values ) {

    // Clear working areas and registries
    cleanupWorkVectors();
    // Reserve space in bad module list
    detids.reserve(100);
  
    // Check if FEDs found in cabling map and event data
    if ( edm::isDebugEnabled() ) {
      if ( cabling.feds().empty() ) {
	edm::LogWarning(sistrip::mlRawToDigi_)
	  << "[sistrip::RawToDigiUnpacker::" << __func__ << "]"
	  << " No FEDs found in cabling map!";
	// Check which FED ids have non-zero size buffers
	std::vector<uint16_t> feds;
	for ( uint16_t ifed = FEDNumbering::MINSiStripFEDID; ifed < FEDNumbering::MAXSiStripFEDID; ifed++ ) {
	  if ( ifed != triggerFedId_ && 
	       buffers.FEDData( static_cast<int>(ifed) ).size() ) {
	    feds.push_back(ifed);
	  }
	}
	LogTrace("Phase2TrackerRawToDigi")
	  << "[sistrip::RawToDigiUnpacker::" << __func__ << "]"
	  << " Found " 
	  << feds.size() 
	  << " FED buffers with non-zero size!";
      }
    }

    // Flag for EventSummary update using DAQ register  
    // bool first_fed = true;
  
    // Retrieve FED ids from cabling map and iterate through 
    std::vector<uint16_t>::const_iterator ifed = cabling.feds().begin();
    for ( ; ifed != cabling.feds().end(); ifed++ ) {

      // ignore trigger FED
      if ( *ifed == triggerFedId_ ) { continue;  }
    
      // Retrieve FED raw data for given FED 
      const FEDRawData& input = buffers.FEDData( static_cast<int>(*ifed) );
    
      // Some debug on FED buffer size
      if ( edm::isDebugEnabled() ) {
	if ( first_ && input.data() ) {
	  std::stringstream ss;
	  ss << "[sistrip::RawToDigiUnpacker::" << __func__ << "]"
	     << " Found FED id " 
	     << std::setw(4) << std::setfill(' ') << *ifed 
	     << " in FEDRawDataCollection"
	     << " with non-zero pointer 0x" 
	     << std::hex
	     << std::setw(8) << std::setfill('0') 
	     << reinterpret_cast<uint32_t*>( const_cast<uint8_t*>(input.data()))
	     << std::dec
	     << " and size " 
	     << std::setw(5) << std::setfill(' ') << input.size()
	     << " chars";
	  LogTrace("Phase2TrackerRawToDigi") << ss.str();
	}	
      }
    
      // Dump of FEDRawData to stdout
      if ( edm::isDebugEnabled() ) {
	if ( fedBufferDumpFreq_ && !(event_%fedBufferDumpFreq_) ) {
	  std::stringstream ss;
	  dumpRawData( *ifed, input, ss );
	  edm::LogVerbatim(sistrip::mlRawToDigi_) << ss.str();
	}
      }
      
      // get the cabling connections for this FED
      const std::vector<FedChannelConnection>& conns = cabling.connections(*ifed);
    
      // Check on FEDRawData pointer
      if ( !input.data() ) {
	if ( edm::isDebugEnabled() ) {
	  edm::LogWarning(sistrip::mlRawToDigi_)
	    << "[sistrip::RawToDigiUnpacker::" << __func__ << "]"
	    << " NULL pointer to FEDRawData for FED id " 
	    << *ifed;
	}
        // Mark FED modules as bad
        detids.reserve(detids.size()+conns.size());
        std::vector<FedChannelConnection>::const_iterator iconn = conns.begin();
        for ( ; iconn != conns.end(); iconn++ ) {
          if ( !iconn->detId() || iconn->detId() == sistrip::invalid32_ ) continue;
          detids.push_back(iconn->detId()); //@@ Possible multiple entries (ok for Giovanni)
        }
	continue;
      }	
    
      // Check on FEDRawData size
      if ( !input.size() ) {
	if ( edm::isDebugEnabled() ) {
	  edm::LogWarning(sistrip::mlRawToDigi_)
	    << "[sistrip::RawToDigiUnpacker::" << __func__ << "]"
	    << " FEDRawData has zero size for FED id " 
	    << *ifed;
	}
        // Mark FED modules as bad
        detids.reserve(detids.size()+conns.size());
        std::vector<FedChannelConnection>::const_iterator iconn = conns.begin();
        for ( ; iconn != conns.end(); iconn++ ) {
          if ( !iconn->detId() || iconn->detId() == sistrip::invalid32_ ) continue;
          detids.push_back(iconn->detId()); //@@ Possible multiple entries (ok for Giovanni)
        }
        continue;
      }
      
      // construct FEDBuffer
      std::auto_ptr<sistrip::Phase2TrackerFEDBuffer> buffer;
      try {
        buffer.reset(new sistrip::Phase2TrackerFEDBuffer(input.data(),input.size()));
        if (!buffer->doChecks()) {
          if (!unpackBadChannels_ || !buffer->checkNoFEOverflows() )
            throw cms::Exception("Phase2TrackerFEDBuffer") << "FED Buffer check fails for FED ID " << *ifed << ".";
        }
        if (doFullCorruptBufferChecks_ && !buffer->doCorruptBufferChecks()) {
          throw cms::Exception("Phase2TrackerFEDBuffer") << "FED corrupt buffer check fails for FED ID " << *ifed << ".";
        }
      }
      catch (const cms::Exception& e) { 
	if ( edm::isDebugEnabled() ) {
	  edm::LogWarning("sistrip::RawToDigiUnpacker") << "Exception caught when creating Phase2TrackerFEDBuffer object for FED " << *ifed << ": " << e.what();
	}
        // FED buffer is bad and should not be unpacked. Skip this FED and mark all modules as bad. 
        std::vector<FedChannelConnection>::const_iterator iconn = conns.begin();
        for ( ; iconn != conns.end(); iconn++ ) {
          if ( !iconn->detId() || iconn->detId() == sistrip::invalid32_ ) continue;
          detids.push_back(iconn->detId()); //@@ Possible multiple entries (ok for Giovanni)
        }
        continue;
      }

/*    // FIXME: commented as Tracker Head as changed on CBC version
      // TODO: is EventSummary needed?
      // commented => sumary is an intup variable of createDigis

      // Check if EventSummary ("trigger FED info") needs updating
      if ( first_fed && useDaqRegister_ ) { updateEventSummary( *buffer, summary ); first_fed = false; }

      // Check to see if EventSummary info is set
      if ( edm::isDebugEnabled() ) {
	if ( !quiet_ && !summary.isSet() ) {
	  std::stringstream ss;
	  ss << "[sistrip::RawToDigiUnpacker::" << __func__ << "]"
	     << " EventSummary is not set correctly!"
	     << " Missing information from both \"trigger FED\" and \"DAQ registers\"!";
	  edm::LogWarning(sistrip::mlRawToDigi_) << ss.str();
	}
      }
    
      // Check to see if event is to be analyzed according to EventSummary
      if ( !summary.valid() ) { 
	if ( edm::isDebugEnabled() ) {
	  LogTrace("Phase2TrackerRawToDigi")
	    << "[sistrip::RawToDigiUnpacker::" << __func__ << "]"
	    << " EventSummary is not valid: skipping...";
	}
	continue; 
      }


      // Retrive run type
   TODO: check if this can be removed useFedKey_ sould be allways false ?
      sistrip::RunType runType_ = summary.runType();
      if( runType_ == sistrip::APV_LATENCY || runType_ == sistrip::FINE_DELAY ) { useFedKey_ = false; } 
*/
      // extract readout mode from CBC tracker header
      sistrip::FEDReadoutMode mode = buffer->readoutMode(); 
     
      // Dump of FED buffer
      if ( edm::isDebugEnabled() ) {
	if ( fedEventDumpFreq_ && !(event_%fedEventDumpFreq_) ) {
	  std::stringstream ss;
	  buffer->dump( ss );
	  edm::LogVerbatim(sistrip::mlRawToDigi_) << ss.str();
	}
      }
    
      // Iterate through FED channels, extract payload and create Digis
      std::vector<FedChannelConnection>::const_iterator iconn = conns.begin();
      for ( ; iconn != conns.end(); iconn++ ) {


	// Check if fed connection is valid
	if ( !iconn->isConnected() ) { continue; }
        
        // Check DetId is valid (if to be used as key)
	if ( !useFedKey_ && ( !iconn->detId() || iconn->detId() == sistrip::invalid32_ ) ) { continue; }
      
	// Check FED channel
        // TODO: implement those method in Phase2TrackerFEDBuffer
/*
	/// FED channel
	uint16_t chan = iconn->fedCh();

	if (!buffer->channelGood(iconn->fedCh(),doAPVEmulatorCheck_)) {
          if (!unpackBadChannels_ || !(buffer->fePresent(iconn->fedCh()/FEDCH_PER_FEUNIT) && buffer->feEnabled(iconn->fedCh()/FEDCH_PER_FEUNIT)) ) {
            detids.push_back(iconn->detId()); //@@ Possible multiple entries (ok for Giovanni)
            continue;
          }
	}
*/

         uint32_t key = iconn->detId();
        /* TODO: determine if key is iconn->detId(); or  iconn->fedCh()
	// Determine whether FED key is inferred from cabling or channel loop
	uint32_t fed_key = ( summary.runType() == sistrip::FED_CABLING ) ? ( ( *ifed & sistrip::invalid_ ) << 16 ) | ( chan & sistrip::invalid_ ) : ( ( iconn->fedId() & sistrip::invalid_ ) << 16 ) | ( iconn->fedCh() & sistrip::invalid_ );

	// Determine whether DetId or FED key should be used to index digi containers
	uint32_t key = ( useFedKey_ || mode == sistrip::READOUT_MODE_SCOPE ) ? fed_key : iconn->detId();
      
#ifndef CBC_RAWTODIGI // {
	// Determine APV std::pair number (needed only when using DetId)
	uint16_t ipair = ( useFedKey_ || mode == sistrip::READOUT_MODE_SCOPE ) ? 0 : iconn->apvPairNumber();
#endif // }
*/

      

// New READOUT_MODE_PROC_RAW and  READOUT_MODE_ZERO_SUPPRESSED for CBC configuration

	if ( mode == sistrip::READOUT_MODE_PROC_RAW)
        { 
          // Strip input data for top and bottom is alternate.
          // Here we split CBC strip data as two independent channels (bottom and top)
          // key for Top is incremented by 1 => Be sure that detCh is not incremental

          // input
          // CBC strips indices input for channel key: 
          // TOP    : 1,3,5,7, ... 253, 255
          // BOTTOM : 0,2,4,6, ... 252, 254

          // Output
          // CBC strips indices for channel key (Bottom): 
          // BOTTOM : 0,1,2,3,4 ... 127
          // CBC strips indices for channel key+1 (Top): 
          // TOP    : 0,1,2,3,4 ... 127

          // TODO: be sure of bottom and top definitions or even, odd
          // create two registers first one for Top second for Bottom 
          // TODO: Size is constant 128 ==> initial position shift is hard coded for top

	  std::vector<SiStripRawDigi> stripBottom; 
	  std::vector<SiStripRawDigi> stripTop; 

          // create Raw unpacker
	  sistrip::Phase2TrackerFEDRawChannelUnpacker unpacker = sistrip::Phase2TrackerFEDRawChannelUnpacker(buffer->channel(iconn->fedCh()));
	  while (unpacker.hasData())
          { // store cluster position and size
            stripBottom.push_back(SiStripRawDigi( static_cast<uint16_t>(unpacker.stripOn()) ));
            unpacker++; // get odd strip
            // quit if it was the last strip
            if (!unpacker.hasData()) break;
            stripTop.push_back(SiStripRawDigi( static_cast<uint16_t>(unpacker.stripOn()) ));
            unpacker++; // get next even strip
          }

          // store Bottom 
	  Registry regItemBottom(key, 0, proc_work_digis_.size(), stripBottom.size());
	  proc_work_registry_.push_back(regItemBottom);
          proc_work_digis_.insert(proc_work_digis_.end(),stripBottom.begin(),stripBottom.end());

          // then Top 
	  Registry regItemTop(key+1, 0, proc_work_digis_.size() , stripTop.size()); 
	  proc_work_registry_.push_back(regItemTop);
          proc_work_digis_.insert(proc_work_digis_.end(),stripTop.begin(),stripTop.end());

        } // end READOUT_MODE_PROC_RAW

	else if (mode == sistrip::READOUT_MODE_ZERO_SUPPRESSED )
        { // create a register and store current index of zs_work_digis_ vector in regItem.index
          Registry regItem(key, 0, zs_work_digis_.size(), 0);

          // create ZS unpacker
          sistrip::Phase2TrackerFEDZSChannelUnpacker unpacker = sistrip::Phase2TrackerFEDZSChannelUnpacker(buffer->channel(iconn->fedCh()));

          while (unpacker.hasData())
          { // store cluster position and size
            zs_work_digis_.push_back(SiStripDigi( static_cast<uint16_t>(unpacker.clusterIndex()),
                                                  static_cast<uint16_t>(unpacker.clusterLength()) ));
            // shift until next cluster
            unpacker++;
          }

          // calculate the length of the vector collection added for this fed channel
          regItem.length = zs_work_digis_.size() - regItem.index;
          if (regItem.length > 0)
          { // store the position of first cluster on this fedchannel
            regItem.first = zs_work_digis_[regItem.index].strip();
            // now keep the info
            zs_work_registry_.push_back(regItem);
          }
        } // end READOUT_MODE_ZERO_SUPPRESSED

	
	else
        { // Unknown readout mode! => assume scope mode
          //TODO: Exception
	} 
      } // channel loop
    } // fed loop

    // bad channels warning
    unsigned int detIdsSize = detids.size();
    if ( edm::isDebugEnabled() && detIdsSize ) {
      std::ostringstream ss;
      ss << "[sistrip::RawToDigiUnpacker::" << __func__ << "]"
         << " Problems were found in data and " << detIdsSize << " channels could not be unpacked. "
         << "See output of FED Hardware monitoring for more information. ";
      edm::LogWarning(sistrip::mlRawToDigi_) << ss.str();
    }
    if( (errorThreshold_ != 0) && (detIdsSize > errorThreshold_) ) {
      edm::LogError("TooManyErrors") << "Total number of errors = " << detIdsSize;
    }

    // update DetSetVectors
    update(scope_mode, virgin_raw, proc_raw, zero_suppr, cm_values);

    // increment event counter
    event_++;
  
    // no longer first event!
    if ( first_ ) { first_ = false; }
  
    // final cleanup, just in case
    cleanupWorkVectors();
  }

  void RawToDigiUnpacker::update( RawDigis& scope_mode, RawDigis& virgin_raw, RawDigis& proc_raw, Digis& zero_suppr, RawDigis& common_mode ) {
  
    if ( ! zs_work_registry_.empty() ) {
      std::sort( zs_work_registry_.begin(), zs_work_registry_.end() );
      std::vector< edm::DetSet<SiStripDigi> > sorted_and_merged;
      sorted_and_merged.reserve(  std::min(zs_work_registry_.size(), size_t(17000)) );
    
      bool errorInData = false;
      std::vector<Registry>::iterator it = zs_work_registry_.begin(), it2 = it+1, end = zs_work_registry_.end();
      while (it < end) {
	sorted_and_merged.push_back( edm::DetSet<SiStripDigi>(it->detid) );
	std::vector<SiStripDigi> & digis = sorted_and_merged.back().data;
	// first count how many digis we have
	size_t len = it->length;
	for (it2 = it+1; (it2 != end) && (it2->detid == it->detid); ++it2) { len += it2->length; }
	// reserve memory 
	digis.reserve(len);
	// push them in
	for (it2 = it+0; (it2 != end) && (it2->detid == it->detid); ++it2) {
	  digis.insert( digis.end(), & zs_work_digis_[it2->index], & zs_work_digis_[it2->index + it2->length] );
	}
	it = it2;
      }
    
      // check sorting
      if (!__gnu_cxx::is_sorted( sorted_and_merged.begin(), sorted_and_merged.end() )) {
	// this is an error in the code: i DID sort it already!
	throw cms::Exception("Bug Found") 
	  << "Container must be already sorted!\nat " 
	  << __FILE__ 
	  << ", line " 
	  << __LINE__ 
	  <<"\n";
      }
    
      std::vector< edm::DetSet<SiStripDigi> >::iterator iii = sorted_and_merged.begin();
      std::vector< edm::DetSet<SiStripDigi> >::iterator jjj = sorted_and_merged.end();
      for ( ; iii != jjj; ++iii ) { 
	if ( ! __gnu_cxx::is_sorted( iii->begin(), iii->end() ) ) {
	  // this might be an error in the data, if the raws from one FED are not sorted
	  iii->clear(); 
	  errorInData = true;
	}
      }	
    
      // output error
      if (errorInData) edm::LogWarning("CorruptData") << "Some modules contained corrupted ZS raw data, and have been skipped in unpacking\n";
    
      // make output DetSetVector
      edm::DetSetVector<SiStripDigi> zero_suppr_dsv( sorted_and_merged, true ); 
      zero_suppr.swap( zero_suppr_dsv );
    } 
  
    // Populate final DetSetVector container with VR data 
    if ( !virgin_work_registry_.empty() ) {

      std::sort( virgin_work_registry_.begin(), virgin_work_registry_.end() );
    
      std::vector< edm::DetSet<SiStripRawDigi> > sorted_and_merged;
      sorted_and_merged.reserve( std::min(virgin_work_registry_.size(), size_t(17000)) );
    
      bool errorInData = false;
      std::vector<Registry>::iterator it = virgin_work_registry_.begin(), it2, end = virgin_work_registry_.end();
      while (it < end) {
	sorted_and_merged.push_back( edm::DetSet<SiStripRawDigi>(it->detid) );
	std::vector<SiStripRawDigi> & digis = sorted_and_merged.back().data;
      
	bool isDetOk = true; 
	// first count how many digis we have
	int maxFirstStrip = it->first;
	for (it2 = it+1; (it2 != end) && (it2->detid == it->detid); ++it2) { 
	  // duplicated APV or data corruption. DO NOT 'break' here!
	  if (it2->first <= maxFirstStrip) { isDetOk = false; continue; } 
	  maxFirstStrip = it2->first;                           
	}
	if (!isDetOk) { errorInData = true; it = it2; continue; } // skip whole det
      
	// make room for 256 * (max_apv_pair + 1) Raw Digis
	digis.resize(maxFirstStrip + 256);
	// push them in
	for (it2 = it+0; (it2 != end) && (it2->detid == it->detid); ++it2) {
	  // data corruption. DO NOT 'break' here
	  if (it->length != 256)  { isDetOk = false; continue; } 
	  std::copy( & virgin_work_digis_[it2->index], & virgin_work_digis_[it2->index + it2->length], & digis[it2->first] );
	}
	if (!isDetOk) { errorInData = true; digis.clear(); it = it2; continue; } // skip whole det
	it = it2;
      }
    
      // output error
      if (errorInData) edm::LogWarning("CorruptData") << "Some modules contained corrupted virgin raw data, and have been skipped in unpacking\n"; 
    
      // check sorting
      if ( !__gnu_cxx::is_sorted( sorted_and_merged.begin(), sorted_and_merged.end()  ) ) {
	// this is an error in the code: i DID sort it already!
	throw cms::Exception("Bug Found") 
	  << "Container must be already sorted!\nat " 
	  << __FILE__ 
	  << ", line " 
	  << __LINE__ 
	  <<"\n";
      }
    
      // make output DetSetVector
      edm::DetSetVector<SiStripRawDigi> virgin_raw_dsv( sorted_and_merged, true ); 
      virgin_raw.swap( virgin_raw_dsv );
    }
  
    // Populate final DetSetVector container with VR data 
    if ( !proc_work_registry_.empty() ) {
      std::sort( proc_work_registry_.begin(), proc_work_registry_.end() );
    
      std::vector< edm::DetSet<SiStripRawDigi> > sorted_and_merged;
      sorted_and_merged.reserve( std::min(proc_work_registry_.size(), size_t(17000)) );
    
      bool errorInData = false;
      std::vector<Registry>::iterator it = proc_work_registry_.begin(), it2, end = proc_work_registry_.end();
      while (it < end) {
	sorted_and_merged.push_back( edm::DetSet<SiStripRawDigi>(it->detid) );
	std::vector<SiStripRawDigi> & digis = sorted_and_merged.back().data;
      
	bool isDetOk = true; 
	// first count how many digis we have
	int maxFirstStrip = it->first;
	for (it2 = it+1; (it2 != end) && (it2->detid == it->detid); ++it2) { 
	  // duplicated APV or data corruption. DO NOT 'break' here!
	  if (it2->first <= maxFirstStrip) { isDetOk = false; continue; } 
	  maxFirstStrip = it2->first;                           
	}
	// skip whole det
	if (!isDetOk) { errorInData = true; it = it2; continue; } 
      
	// make room for 256 * (max_apv_pair + 1) Raw Digis
	digis.resize(maxFirstStrip + 256);
	// push them in
	for (it2 = it+0; (it2 != end) && (it2->detid == it->detid); ++it2) {
	  // data corruption. DO NOT 'break' here
	  if (it->length != 256)  { isDetOk = false; continue; } 
	  std::copy( & proc_work_digis_[it2->index], & proc_work_digis_[it2->index + it2->length], & digis[it2->first] );
	}
	// skip whole det
	if (!isDetOk) { errorInData = true; digis.clear(); it = it2; continue; } 
	it = it2;
      }
    
      // output error
      if (errorInData) edm::LogWarning("CorruptData") << "Some modules contained corrupted proc raw data, and have been skipped in unpacking\n";
    
      // check sorting
      if ( !__gnu_cxx::is_sorted( sorted_and_merged.begin(), sorted_and_merged.end()  ) ) {
	// this is an error in the code: i DID sort it already!
	throw cms::Exception("Bug Found") 
	  << "Container must be already sorted!\nat " 
	  << __FILE__ 
	  << ", line " 
	  << __LINE__ 
	  <<"\n";
      }
    
      // make output DetSetVector
      edm::DetSetVector<SiStripRawDigi> proc_raw_dsv( sorted_and_merged, true ); 
      proc_raw.swap( proc_raw_dsv );
    }
  
    // Populate final DetSetVector container with SM data 
    if ( !scope_work_registry_.empty() ) {
      std::sort( scope_work_registry_.begin(), scope_work_registry_.end() );
    
      std::vector< edm::DetSet<SiStripRawDigi> > sorted_and_merged;
      sorted_and_merged.reserve( scope_work_registry_.size() );
    
      bool errorInData = false;
      std::vector<Registry>::iterator it, end;
      for (it = scope_work_registry_.begin(), end = scope_work_registry_.end() ; it != end; ++it) {
	sorted_and_merged.push_back( edm::DetSet<SiStripRawDigi>(it->detid) );
	std::vector<SiStripRawDigi> & digis = sorted_and_merged.back().data;
	digis.insert( digis.end(), & scope_work_digis_[it->index], & scope_work_digis_[it->index + it->length] );
      
	if ( (it +1 != end) && (it->detid == (it+1)->detid) ) {
	  errorInData = true; 
	  // let's skip *all* the detsets for that key, as we don't know which is the correct one!
	  do { ++it; } while ( ( it+1 != end) && (it->detid == (it+1)->detid) );
	}
      }

      // output error
      if (errorInData) edm::LogWarning("CorruptData") << "Some fed keys contained corrupted scope mode data, and have been skipped in unpacking\n"; 

      // check sorting
      if ( !__gnu_cxx::is_sorted( sorted_and_merged.begin(), sorted_and_merged.end()  ) ) {
	// this is an error in the code: i DID sort it already!
	throw cms::Exception("Bug Found") 
	  << "Container must be already sorted!\nat " 
	  << __FILE__ 
	  << ", line " 
	  << __LINE__ 
	  <<"\n";
      }

      // make output DetSetVector
      edm::DetSetVector<SiStripRawDigi> scope_mode_dsv( sorted_and_merged, true ); 
      scope_mode.swap( scope_mode_dsv );
    }

    // Populate DetSetVector with Common Mode values 
    if ( extractCm_ ) {

      // Populate final DetSetVector container with VR data 
      if ( !cm_work_registry_.empty() ) {

	std::sort( cm_work_registry_.begin(), cm_work_registry_.end() );
    
	std::vector< edm::DetSet<SiStripRawDigi> > sorted_and_merged;
	sorted_and_merged.reserve( std::min(cm_work_registry_.size(), size_t(17000)) );
    
	bool errorInData = false;
	std::vector<Registry>::iterator it = cm_work_registry_.begin(), it2, end = cm_work_registry_.end();
	while (it < end) {
	  sorted_and_merged.push_back( edm::DetSet<SiStripRawDigi>(it->detid) );
	  std::vector<SiStripRawDigi> & digis = sorted_and_merged.back().data;
      
	  bool isDetOk = true; 
	  // first count how many digis we have
	  int maxFirstStrip = it->first;
	  for (it2 = it+1; (it2 != end) && (it2->detid == it->detid); ++it2) { 
	    // duplicated APV or data corruption. DO NOT 'break' here!
	    if (it2->first <= maxFirstStrip) { isDetOk = false; continue; } 
	    maxFirstStrip = it2->first;                           
	  }
	  if (!isDetOk) { errorInData = true; it = it2; continue; } // skip whole det
	  
	  // make room for 2 * (max_apv_pair + 1) Common mode values
	  digis.resize(maxFirstStrip + 2);
	  // push them in
	  for (it2 = it+0; (it2 != end) && (it2->detid == it->detid); ++it2) {
	    // data corruption. DO NOT 'break' here
	    if (it->length != 2)  { isDetOk = false; continue; } 
	    std::copy( & cm_work_digis_[it2->index], & cm_work_digis_[it2->index + it2->length], & digis[it2->first] );
	  }
	  if (!isDetOk) { errorInData = true; digis.clear(); it = it2; continue; } // skip whole det
	  it = it2;
	}
    
	// output error
	if (errorInData) edm::LogWarning("CorruptData") << "Some modules contained corrupted common mode data, and have been skipped in unpacking\n"; 
    
	// check sorting
	if ( !__gnu_cxx::is_sorted( sorted_and_merged.begin(), sorted_and_merged.end()  ) ) {
	  // this is an error in the code: i DID sort it already!
	  throw cms::Exception("Bug Found") 
	    << "Container must be already sorted!\nat " 
	    << __FILE__ 
	    << ", line " 
	    << __LINE__ 
	    <<"\n";
	}
    
	// make output DetSetVector
	edm::DetSetVector<SiStripRawDigi> common_mode_dsv( sorted_and_merged, true ); 
	common_mode.swap( common_mode_dsv );
      }
      
    }
    
  }
  
  void RawToDigiUnpacker::cleanupWorkVectors() {
    // Clear working areas and registries
    zs_work_registry_.clear();      zs_work_digis_.clear();
    virgin_work_registry_.clear();  virgin_work_digis_.clear();
    proc_work_registry_.clear();    proc_work_digis_.clear();
    scope_work_registry_.clear();   scope_work_digis_.clear();
    cm_work_registry_.clear();      cm_work_digis_.clear();
  }

  /*
  void RawToDigiUnpacker::triggerFed( const FEDRawDataCollection& buffers, SiStripEventSummary& summary, const uint32_t& event ) {
  
    // Pointer to data (recast as 32-bit words) and number of 32-bit words
    uint32_t* data_u32 = 0;
    uint32_t  size_u32 = 0;
  
    // Search mode
    if ( triggerFedId_ < 0 ) { 
    
      uint16_t ifed = 0;
      while ( triggerFedId_ < 0 && 
	      ifed < 1 + FEDNumbering::lastFEDId() ) {
	const FEDRawData& trigger_fed = buffers.FEDData( ifed );
	if ( trigger_fed.data() && trigger_fed.size() ) {
	  uint8_t*  temp = const_cast<uint8_t*>( trigger_fed.data() );
	  data_u32 = reinterpret_cast<uint32_t*>( temp ) + sizeof(fedh_t)/sizeof(uint32_t) + 1;
	  size_u32 = trigger_fed.size()/sizeof(uint32_t) - sizeof(fedh_t)/sizeof(uint32_t) - 1;
	  fedt_t* fed_trailer = reinterpret_cast<fedt_t*>( temp + trigger_fed.size() - sizeof(fedt_t) );
	  if ( fed_trailer->conscheck == 0xDEADFACE ) { 
	    triggerFedId_ = ifed; 
	    if ( edm::isDebugEnabled() ) {
	      std::stringstream ss;
	      ss << "[sistrip::RawToDigiUnpacker::" << __func__ << "]"
		 << " Search mode for 'trigger FED' activated!"
		 << " Found 'trigger FED' info with id " << triggerFedId_;
	      LogTrace("Phase2TrackerRawToDigi") << ss.str();
	    }
	  }
	}
	ifed++;
      }
      if ( triggerFedId_ < 0 ) {
	triggerFedId_ = 0;
	if ( edm::isDebugEnabled() ) {
	  std::stringstream ss;
	  ss << "[sistrip::RawToDigiUnpacker::" << __func__ << "]"
	     << " Search mode for 'trigger FED' activated!"
	     << " 'Trigger FED' info not found!";
	  edm::LogWarning(sistrip::mlRawToDigi_) << ss.str();
	}
      }  
    } 

    // "Trigger FED" id given in .cfg file
    else if ( triggerFedId_ > 0 ) { 
    
      const FEDRawData& trigger_fed = buffers.FEDData( triggerFedId_ );
      if ( trigger_fed.data() && trigger_fed.size() ) {
	uint8_t*  temp = const_cast<uint8_t*>( trigger_fed.data() );
	data_u32 = reinterpret_cast<uint32_t*>( temp ) + sizeof(fedh_t)/sizeof(uint32_t) + 1;
	size_u32 = trigger_fed.size()/sizeof(uint32_t) - sizeof(fedh_t)/sizeof(uint32_t) - 1;
	fedt_t* fed_trailer = reinterpret_cast<fedt_t*>( temp + trigger_fed.size() - sizeof(fedt_t) );
	if ( fed_trailer->conscheck != 0xDEADFACE ) { 
	  if ( edm::isDebugEnabled() ) {
	    edm::LogWarning(sistrip::mlRawToDigi_) 
	      << "[sistrip::RawToDigiUnpacker::" << __func__ << "]"
	      << " Unexpected stamp found in DAQ trailer (ie, not 0xDEADFACE)!"
	      << " Buffer appears not to contain 'trigger FED' data!";
	  }
	  triggerFedId_ = 0; 
	}
      }
      
    } else { 
      triggerFedId_ = 0; 
      data_u32 = 0;
      size_u32 = 0;
    }
  
    // Populate summary object with commissioning information
    if ( triggerFedId_ > 0 ) { 

      // Some checks
      if ( !data_u32 ) {
	if ( edm::isDebugEnabled() ) {
	  std::stringstream ss;
	  ss << "[sistrip::RawToDigiUnpacker::" << __func__ << "]"
	     << " NULL pointer to 'trigger FED' data";
	  edm::LogWarning(sistrip::mlRawToDigi_) << ss.str();
	}
	return;
      } 
      if ( size_u32 < sizeof(TFHeaderDescription)/sizeof(uint32_t) ) {
	if ( edm::isDebugEnabled() ) {
	  std::stringstream ss;
	  ss << "[sistrip::RawToDigiUnpacker::" << __func__ << "]"
	     << " Unexpected 'Trigger FED' data size [32-bit words]: " << size_u32;
	  edm::LogWarning(sistrip::mlRawToDigi_) << ss.str();
	}
	return;
      }
    
      // Write event-specific data to event
      TFHeaderDescription* header = (TFHeaderDescription*) data_u32;
      summary.event( static_cast<uint32_t>( header->getFedEventNumber()) );
      summary.bx( static_cast<uint32_t>( header->getBunchCrossing()) );
    
      // Write commissioning information to event 
      uint32_t hsize = sizeof(TFHeaderDescription)/sizeof(uint32_t);
      uint32_t* head = &data_u32[hsize];
      summary.commissioningInfo( head, event );
      summary.triggerFed( triggerFedId_ );
    
    }

    // Some debug
    if ( summary.isSet() && once_ ) {
      if ( edm::isDebugEnabled() ) {
	std::stringstream ss;
	ss << "[sistrip::RawToDigiUnpacker::" << __func__ << "]"
	   << " EventSummary built from \"trigger FED\":" 
	   << std::endl << summary;
	LogTrace("Phase2TrackerRawToDigi") << ss.str();
      }
      once_ = false;
    }
  }
  */

  // XXX: not used?
  void RawToDigiUnpacker::locateStartOfFedBuffer( const uint16_t& fed_id, const FEDRawData& input, FEDRawData& output ) {
  
    // Check size of input buffer
    if ( input.size() < 24 ) { 
      output.resize( input.size() ); // Return UNadjusted buffer start position and size
      memcpy( output.data(), input.data(), input.size() );
      if ( edm::isDebugEnabled() ) {
	std::stringstream ss; 
	ss << "[sistrip::RawToDigiUnpacker::" << __func__ << "] "
	   << "Input FEDRawData with FED id " << fed_id 
	   << " has size " << input.size();
	edm::LogWarning(sistrip::mlRawToDigi_) << ss.str();
      }
      return;
    } 
  
    // Iterator through buffer to find DAQ header 
    bool found = false;
    uint16_t ichar = 0;
    while ( ichar < input.size()-16 && !found ) {
      uint16_t offset = headerBytes_ < 0 ? ichar : headerBytes_; // Negative value means use "search mode" to find DAQ header
      uint32_t* input_u32   = reinterpret_cast<uint32_t*>( const_cast<unsigned char*>( input.data() ) + offset );
      uint32_t* fed_trailer = reinterpret_cast<uint32_t*>( const_cast<unsigned char*>( input.data() ) + input.size() - 8 );
    
      // see info on FED 32-bit swapping at end-of-file

      bool old_vme_header = ( input_u32[0] & 0xF0000000 ) == 0x50000000 && ( fed_trailer[0]  & 0xF0000000 ) == 0xA0000000 && ( (fed_trailer[0] & 0x00FFFFFF)*0x8 ) == (input.size() - offset);
    
      bool old_slink_header = ( input_u32[1] & 0xF0000000 ) == 0x50000000 && ( fed_trailer[1]  & 0xF0000000 ) == 0xA0000000 && ( (fed_trailer[1] & 0x00FFFFFF)*0x8 ) == (input.size() - offset);
    
      bool old_slink_payload = ( input_u32[3] & 0xFF000000 ) == 0xED000000;
    
      bool new_buffer_format = ( input_u32[2] & 0xFF000000 ) == 0xC5000000;
    
      if ( old_vme_header )  {
      
	// Found DAQ header at byte position 'offset'
	found = true;
	output.resize( input.size()-offset );
	memcpy( output.data(),         // target
		input.data()+offset,   // source
		input.size()-offset ); // nbytes
	if ( headerBytes_ < 0 ) {
	  if ( edm::isDebugEnabled() ) {
	    std::stringstream ss;
	    ss << "[sistrip::RawToDigiUnpacker::" << __func__ << "]" 
	       << " Buffer for FED id " << fed_id 
	       << " has been found at byte position " << offset
	       << " with a size of " << input.size()-offset << " bytes."
	       << " Adjust the configurable 'AppendedBytes' to " << offset;
	    LogTrace("Phase2TrackerRawToDigi") << ss.str();
	  }
	}
      
      } else if ( old_slink_header ) {
      
	if ( old_slink_payload ) {
      
	  // Found DAQ header (with MSB and LSB 32-bit words swapped) at byte position 'offset' 
	  found = true;
	  output.resize( input.size()-offset );
	  uint32_t* output_u32 = reinterpret_cast<uint32_t*>( const_cast<unsigned char*>( output.data() ) );
	  uint16_t iter = offset; 
	  while ( iter < output.size() / sizeof(uint32_t) ) {
	    output_u32[iter] = input_u32[iter+1];
	    output_u32[iter+1] = input_u32[iter];
	    iter+=2;
	  }
	  if ( headerBytes_ < 0 ) {
	    if ( edm::isDebugEnabled() ) {
	      std::stringstream ss;
	      ss << "[sistrip::RawToDigiUnpacker::" << __func__ << "]" 
		 << " Buffer (with MSB and LSB 32-bit words swapped) for FED id " << fed_id 
		 << " has been found at byte position " << offset
		 << " with a size of " << output.size() << " bytes."
		 << " Adjust the configurable 'AppendedBytes' to " << offset;
	      LogTrace("Phase2TrackerRawToDigi") << ss.str();
	    }
	  }

	} else if ( new_buffer_format ) {
	
	  // Found DAQ header at byte position 'offset'
	  found = true;
	  output.resize( input.size()-offset );
	  memcpy( output.data(),         // target
		  input.data()+offset,   // source
		  input.size()-offset ); // nbytes
	  if ( headerBytes_ < 0 ) {
	    if ( edm::isDebugEnabled() ) {
	      std::stringstream ss;
	      ss << "[sistrip::RawToDigiUnpacker::" << __func__ << "]" 
		 << " Buffer for FED id " << fed_id 
		 << " has been found at byte position " << offset
		 << " with a size of " << input.size()-offset << " bytes."
		 << " Adjust the configurable 'AppendedBytes' to " << offset;
	      LogTrace("Phase2TrackerRawToDigi") << ss.str();
	    }
	  }
	
	} else { headerBytes_ < 0 ? found = false : found = true; }
      } else { headerBytes_ < 0 ? found = false : found = true; }
      ichar++;
    }      
  
    // Check size of output buffer
    if ( output.size() == 0 ) { 
    
      // Did not find DAQ header after search => return buffer with null size
      output.resize( 0 ); //@@ NULL SIZE
      memcpy( output.data(), input.data(), 0 ); //@@ NULL SIZE
      if ( edm::isDebugEnabled() ) {
	std::stringstream ss;
	if ( headerBytes_ < 0 ) {
	  ss << "[sistrip::RawToDigiUnpacker::" << __func__ << "]"
	     << " DAQ header not found within buffer for FED id: " << fed_id;
	} else {
	  uint32_t* input_u32 = reinterpret_cast<uint32_t*>( const_cast<unsigned char*>( input.data() ) );
	  ss << "[sistrip::RawToDigiUnpacker::" << __func__ << "]"
	     << " DAQ header not found at expected location for FED id: " << fed_id << std::endl
	     << " First 64-bit word of buffer is 0x"
	     << std::hex 
	     << std::setfill('0') << std::setw(8) << input_u32[0] 
	     << std::setfill('0') << std::setw(8) << input_u32[1] 
	     << std::dec << std::endl
	     << " Adjust 'AppendedBytes' configurable to '-1' to activate 'search mode'";
	}
	edm::LogWarning(sistrip::mlRawToDigi_) << ss.str();
      }
    
    } else if ( output.size() < 24 ) { // Found DAQ header after search, but too few words
    
      if ( edm::isDebugEnabled() ) {
	std::stringstream ss; 
	ss << "[sistrip::RawToDigiUnpacker::" << __func__ << "]"
	   << " Unexpected buffer size! FEDRawData with FED id " << fed_id 
	   << " has size " << output.size();
	edm::LogWarning(sistrip::mlRawToDigi_) << ss.str();
      }
    }   
  }

/*    // FIXME: commented as Tracker Head as changed on CBC version
  void RawToDigiUnpacker::updateEventSummary( const sistrip::FEDBuffer& fed, SiStripEventSummary& summary ) {
  
    // Retrieve contents of DAQ registers

    sistrip::FEDDAQEventType readout_mode = fed.daqEventType(); 
    uint32_t daq1 = sistrip::invalid32_;
    uint32_t daq2 = sistrip::invalid32_;

    if (fed.headerType() == sistrip::HEADER_TYPE_FULL_DEBUG) {
      const sistrip::FEDFullDebugHeader* header = 0;
      header = dynamic_cast<const sistrip::FEDFullDebugHeader*>(fed.feHeader());
      daq1 = static_cast<uint32_t>( header->daqRegister() ); 
      daq2 = static_cast<uint32_t>( header->daqRegister2() ); 
    }

  
    // If FED DAQ registers contain info, update (and possibly overwrite) EventSummary 
    if ( daq1 != 0 && daq1 != sistrip::invalid32_ ) {
    
      summary.triggerFed( triggerFedId_ );
      summary.fedReadoutMode( readout_mode );
      summary.commissioningInfo( daq1, daq2 );
    
      if ( summary.isSet() && once_ ) {
	if ( edm::isDebugEnabled() ) {
	  std::stringstream ss;
	  ss << "[sistrip::RawToDigiUnpacker::" << __func__ << "]"
	     << " EventSummary built from FED DAQ registers:"
	     << std::endl << summary;
	  LogTrace("Phase2TrackerRawToDigi") << ss.str();
	}
	once_ = false;
      }
    }
  }
*/

  void RawToDigiUnpacker::dumpRawData( uint16_t fed_id, const FEDRawData& buffer, std::stringstream& ss ) {

    ss << "[sistrip::RawToDigiUnpacker::" << __func__ << "]"
       << " Dump of buffer for FED id " <<  fed_id << std::endl
       << " Buffer contains " << buffer.size()
       << " bytes (NB: payload is byte-swapped)" << std::endl;
    uint32_t* buffer_u32 = reinterpret_cast<uint32_t*>( const_cast<unsigned char*>( buffer.data() ) );
    unsigned int empty = 0;

    if ( 0 ) { 

      ss << "Byte->   4 5 6 7 0 1 2 3\n";
      for ( uint32_t i = 0; i < buffer.size()/8; i++ ) {
	unsigned int temp0 = buffer_u32[i*2] & sistrip::invalid32_;
	unsigned int temp1 = buffer_u32[i*2+1] & sistrip::invalid32_;
	if ( !temp0 && !temp1 ) { empty++; }
	else { 
	  if ( empty ) { 
	    ss << "        [ empty  words ]" << std::endl; 
	    empty = 0; 
	  }
	  ss << std::dec
	     << std::setfill(' ')  << std::setw(6) << i*8 << ": " 
	     << std::hex 
	     << std::setfill('0') << std::setw(8) << temp0 
	     << std::setfill('0') << std::setw(8) << temp1 
	     << std::dec
	     << std::endl;
	}
      }

    } else {
    
      ss << "  Byte |  <---- Byte order ----<  | Byte" << std::endl;
      ss << "  cntr |  7  6  5  4  3  2  1  0  | cntr" << std::endl;
      for ( uint32_t i = 0; i < buffer.size()/8; i++ ) {
	//if ( i>=20 && ((i+4)<(buffer.size()/8)) ) { continue; }
	uint16_t tmp0 = buffer.data()[i*8+0] & 0xFF;
	uint16_t tmp1 = buffer.data()[i*8+1] & 0xFF;
	uint16_t tmp2 = buffer.data()[i*8+2] & 0xFF;
	uint16_t tmp3 = buffer.data()[i*8+3] & 0xFF;
	uint16_t tmp4 = buffer.data()[i*8+4] & 0xFF;
	uint16_t tmp5 = buffer.data()[i*8+5] & 0xFF;
	uint16_t tmp6 = buffer.data()[i*8+6] & 0xFF;
	uint16_t tmp7 = buffer.data()[i*8+7] & 0xFF;
// 	if ( !tmp0 && !tmp1 && !tmp2 && !tmp3 &&
// 	     !tmp4 && !tmp5 && !tmp6 && !tmp7 ) { empty++; }
// 	else { 
// 	  if ( empty ) { 
// 	    ss << "         [.." 
// 	       << std::dec << std::setfill('.') << std::setw(4) << empty 
// 	       << " null words....]" << std::endl; 
// 	    empty = 0; 
// 	  }
	  ss << std::dec
	     << std::setfill(' ')  << std::setw(6) << i*8+7 << " : " 
	     << std::hex 
	     << std::setfill('0') << std::setw(2) << tmp7 << " " 
	     << std::setfill('0') << std::setw(2) << tmp6 << " " 
	     << std::setfill('0') << std::setw(2) << tmp5 << " " 
	     << std::setfill('0') << std::setw(2) << tmp4 << " " 
	     << std::setfill('0') << std::setw(2) << tmp3 << " " 
	     << std::setfill('0') << std::setw(2) << tmp2 << " " 
	     << std::setfill('0') << std::setw(2) << tmp1 << " " 
	     << std::setfill('0') << std::setw(2) << tmp0 
	     << std::dec
	     << " :" << std::setfill(' ')  << std::setw(6) << i*8 
	     << std::endl;
// 	}
      }

    }
    ss << "[sistrip::RawToDigiUnpacker::" << __func__ << "]"
       << " End of FED buffer";
  }

  void RawToDigiUnpacker::handleException( std::string method_name, std::string extra_info ) { 

    method_name = "sistrip::RawToDigiUnpacker::" + method_name;
    try {
      throw; // rethrow caught exception to be dealt with below
    } 
    catch ( const cms::Exception& e ) { 
      //throw e; // rethrow cms::Exception to be caught by framework
    }
    catch ( const std::exception& e ) {
      if ( edm::isDebugEnabled() ) {
	std::stringstream ss;
	ss << "[sistrip::RawToDigiUnpacker::" << __func__ << "]"
	   << " Caught std::exception!" << std::endl;
	if ( extra_info != "" ) { 
	  ss << " Information: " << extra_info << std::endl;
	}
	ss << " Caught std::exception in ["
	   << method_name << "] with message:" << std::endl 
	   << e.what();
	edm::LogWarning(sistrip::mlRawToDigi_) << ss.str();
      }
      //throw cms::Exception(sistrip::mlRawToDigi_) << ss.str();
    }
    catch (...) {
      if ( edm::isDebugEnabled() ) {
	std::stringstream ss;
	ss << "[sistrip::RawToDigiUnpacker::" << __func__ << "]"
	   << " Caught unknown exception!" << std::endl;
	if ( extra_info != "" ) { 
	  ss << " Information: " << extra_info << std::endl;
	}
	ss << "Caught unknown exception in ["
	   << method_name << "]" << std::endl;
	edm::LogWarning(sistrip::mlRawToDigi_) << ss.str();
      }
      //throw cms::Exception(sistrip::mlRawToDigi_) << ss.str();
    }
  }

}

/*
  
Some info on FED buffer 32-bit word swapping. 

Table below indicates if data are swapped relative to the "old"
VME format (as originally expected by the Fed9UEvent class).

-------------------------------------------
| SWAPPED?    |         DATA FORMAT       |
| (wrt "OLD") | OLD (0xED)  | NEW (0xC5)  |
|             | VME | SLINK | VME | SLINK |
-------------------------------------------
| DAQ HEADER  |  N  |   Y   |  Y  |   Y   |
| TRK HEADER  |  N  |   Y   |  N  |   N   |
| PAYLOAD     |  N  |   Y   |  N  |   N   |
| DAQ TRAILER |  N  |   Y   |  Y  |   Y   |
-------------------------------------------

So, in code, we check in code order of bytes in DAQ header/trailer only:
-> if "old_vme_header",           then old format read out via vme, so do nothing.
-> else if "old_slink_header",    then data may be wrapped, so check additionally the TRK header:
---> if "old_slink_payload",       then old format read out via slink, so swap all data;
---> else if "new_buffer_format",  then new format, handled internally by Fed9UEvent, so do nothing.

Pattern matching to find DAQ and tracker headers, and DAQ trailer:
DAQ header,  4 bits, in field  |BOE_1|      with value 0x5
DAQ trailer, 4 bits, in field  |EOE_1|      with value 0xA
TRK header,  8 bits, in field  |Hdr format| with value 0xED or 0xC5

-------------------------------------------------------------------------------------------
| SWAPPED?    |                                 DATA FORMAT                               |
| (wrt "OLD") |               OLD (0xED)            |               NEW (0xC5)            |
|             |       VME        |      SLINK       |       VME        |      SLINK       |
-------------------------------------------------------------------------------------------
| DAQ HEADER  | ........5....... | 5............... | 5............... | 5............... |
| TRK HEADER  | ........ED...... | ED.............. | ........C5...... | ........C5...... |
| PAYLOAD     | ..........EA.... | ..EA............ | ..EA............ | ............EA.. | 
| DAQ TRAILER | ........A....... | A............... | A............... | A............... |
-------------------------------------------------------------------------------------------

*/
