#include "EventFilter/SiStripRawToDigi/interface/SiStripRawToDigiUnpacker.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/src/fed_header.h"
#include "DataFormats/FEDRawData/src/fed_trailer.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
//#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"
#include "DataFormats/SiStripCommon/interface/SiStripEventSummary.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "EventFilter/SiStripRawToDigi/interface/TFHeaderDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Utilities/Timing/interface/TimingReport.h"
#include "Fed9UUtils.hh"
#include "ICException.hh"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <ext/algorithm>

using namespace sistrip;

// -----------------------------------------------------------------------------
/** */
SiStripRawToDigiUnpacker::SiStripRawToDigiUnpacker( int16_t appended_bytes, 
						    int16_t fed_buffer_dump_freq, 
						    int16_t fed_event_dump_freq, 
						    int16_t trigger_fed_id,
						    bool using_fed_key  ) :
  headerBytes_( appended_bytes ),
  fedBufferDumpFreq_( fed_buffer_dump_freq ),
  fedEventDumpFreq_( fed_event_dump_freq ),
  triggerFedId_( trigger_fed_id ),
  useFedKey_( using_fed_key ),
  fedEvent_(0),
  event_(0),
  once_(true),
  first_(true),
  quiet_(true)
{
  if ( edm::isDebugEnabled() ) {
    LogTrace(mlRawToDigi_)
      << "[SiStripRawToDigiUnpacker::"<<__func__<<"]"
      <<" Constructing object...";
  }
  fedEvent_ = new Fed9U::Fed9UEvent();
}

// -----------------------------------------------------------------------------
/** */
SiStripRawToDigiUnpacker::~SiStripRawToDigiUnpacker() {
  if ( edm::isDebugEnabled() ) {
    LogTrace(mlRawToDigi_)
      << "[SiStripRawToDigiUnpacker::"<<__func__<<"]"
      << " Destructing object...";
  }
  if ( fedEvent_ ) { delete fedEvent_; }
}

// -----------------------------------------------------------------------------
/** */
void SiStripRawToDigiUnpacker::createDigis( const SiStripFedCabling& cabling,
					    const FEDRawDataCollection& buffers,
					    SiStripEventSummary& summary,
					    RawDigis& scope_mode,
					    RawDigis& virgin_raw,
					    RawDigis& proc_raw,
					    Digis& zero_suppr ) {
  // Clear working areas and registries
  cleanupWorkVectors();
  
  // Check if FEDs found in cabling map and event data
  if ( edm::isDebugEnabled() ) {
    if ( cabling.feds().empty() ) {
      edm::LogWarning(mlRawToDigi_)
	<< "[SiStripRawToDigiUnpacker::" << __func__ << "]"
	<< " No FEDs found in cabling map!";
      // Check which FED ids have non-zero size buffers
      std::pair<int,int> fed_range = FEDNumbering::getSiStripFEDIds();
      std::vector<uint16_t> feds;
      for ( uint16_t ifed = static_cast<uint16_t>(fed_range.first);
	    ifed < static_cast<uint16_t>(fed_range.second); ifed++ ) {
	if ( ifed != triggerFedId_ && 
	     buffers.FEDData( static_cast<int>(ifed) ).size() ) {
	  feds.push_back(ifed);
	}
      }
      LogTrace(mlRawToDigi_)
	<< "[SiStripRawToDigiUnpacker::" << __func__ << "]"
	<< " Found " << feds.size() << " FED buffers with non-zero size!";
    }
  }

  // Flag for EventSummary update using DAQ register  
  bool first_fed = true;
  
  // Retrieve FED ids from cabling map and iterate through 
  std::vector<uint16_t>::const_iterator ifed = cabling.feds().begin();
  for ( ; ifed != cabling.feds().end(); ifed++ ) {

    //if ( *ifed == triggerFedId_ ) { continue; }
    
    // Retrieve FED raw data for given FED 
    const FEDRawData& input = buffers.FEDData( static_cast<int>(*ifed) );
    
    // Some debug on FED buffer size
    if ( edm::isDebugEnabled() ) {
      if ( first_ && input.data() ) {
	std::stringstream ss;
	ss << "[SiStripRawToDigiUnpacker::" << __func__ << "]"
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
	LogTrace(mlRawToDigi_) << ss.str();
      }	
    }
    
    // Dump of FEDRawData to stdout
    if ( edm::isDebugEnabled() ) {
      if ( fedBufferDumpFreq_ && !(event_%fedBufferDumpFreq_) ) {
	std::stringstream ss;
	dumpRawData( *ifed, input, ss );
	edm::LogVerbatim(mlRawToDigi_) << ss.str();
      }
    }
    
    // Handle 32-bit swapped data (and locate start of FED buffer within raw data)
    FEDRawData output; 
    locateStartOfFedBuffer( *ifed, input, output );
    
    // Recast data to suit Fed9UEvent
    Fed9U::u32* data_u32 = reinterpret_cast<Fed9U::u32*>( const_cast<unsigned char*>( output.data() ) );
    Fed9U::u32  size_u32 = static_cast<Fed9U::u32>( output.size() / 4 ); 

    // Check on FEDRawData pointer
    if ( !data_u32 ) {
      if ( edm::isDebugEnabled() ) {
	edm::LogWarning(mlRawToDigi_)
	  << "[SiStripRawToDigiUnpacker::" << __func__ << "]"
	  << " NULL pointer to FEDRawData for FED id " << *ifed;
      }
      continue;
    }	
    
    // Check on FEDRawData size
    if ( !size_u32 ) {
      if ( edm::isDebugEnabled() ) {
	edm::LogWarning(mlRawToDigi_)
	  << "[SiStripRawToDigiUnpacker::" << __func__ << "]"
	  << " FEDRawData has zero size for FED id " << *ifed;
      }
      continue;
    }
    
    // Initialise Fed9UEvent using present FED buffer
    try {
      fedEvent_->Init( data_u32, 0, size_u32 ); 
      //fedEvent_->checkEvent();
    } catch(...) { 
      handleException( __func__, "Problem when creating Fed9UEvent" ); 
      continue;
    } 
   
    // Check if EventSummary ("trigger FED info") needs updating
    if ( first_fed ) {
      //updateEventSummary( fedEvent_, summary ); //@@ temporarily suppress statements from SiStripEventSummary!
      first_fed = false;
    }
    
    // Check to see if EventSummary info is set
    if ( edm::isDebugEnabled() ) {
      if ( !quiet_ && !summary.isSet() ) {
	std::stringstream ss;
	ss << "[SiStripRawToDigiUnpacker::" << __func__ << "]"
	   << " EventSummary is not set correctly!"
	   << " Missing information from both \"trigger FED\" and \"DAQ registers\"!";
	edm::LogWarning(mlRawToDigi_) << ss.str();
      }
    }
    
    // Check to see if event is to be analyzed according to EventSummary
    if ( !summary.valid() ) { 
      if ( edm::isDebugEnabled() ) {
	LogTrace(mlRawToDigi_)
	  << "[SiStripRawToDigiUnpacker::" << __func__ << "]"
	  << " EventSummary is not valid: skipping...";
      }
      continue; 
    }
    
    // Retrive readout mode
    sistrip::FedReadoutMode mode = sistrip::UNDEFINED_FED_READOUT_MODE;
    try {
      mode = fedReadoutMode( static_cast<unsigned int>( fedEvent_->getSpecialTrackerEventType() ) );
    } catch(...) { handleException( __func__, "Problem extracting readout mode from Fed9UEvent" ); } 
    
    // Retrive run type
    sistrip::RunType runType_ = summary.runType();
    if( runType_ == sistrip::APV_LATENCY || runType_ == sistrip::FINE_DELAY ) { useFedKey_ = false; } //@@ force!
     
    // Dump of FED buffer
    if ( edm::isDebugEnabled() ) {
      if ( fedEventDumpFreq_ && !(event_%fedEventDumpFreq_) ) {
	std::stringstream ss;
	fedEvent_->dump( ss );
	edm::LogVerbatim(mlRawToDigi_) << ss.str();
      }
    }
    
    // Iterate through FED channels, extract payload and create Digis
    const std::vector<FedChannelConnection>& conns = cabling.connections(*ifed);
    std::vector<FedChannelConnection>::const_iterator iconn = conns.begin();
    for ( ; iconn != conns.end(); iconn++ ) {

      // Check if FedId is valid
      if ( !iconn->isConnected() ) { continue; }
      
      // Check DetId is valid (if to be used as key)
      if ( !useFedKey_ && ( !iconn->detId() || iconn->detId() == sistrip::invalid32_ ) ) { continue; }

      // Retrieve channel using Fed9UAddress 
      uint16_t channel = iconn->fedCh();
      uint16_t iunit = channel / 12;
      uint16_t ichan = channel % 12;
      uint16_t chan  = 12 * iunit + ichan;
      try {
	Fed9U::Fed9UAddress addr;
	addr.setFedChannel( static_cast<unsigned char>( channel ) );
	iunit = addr.getFedFeUnit();
	ichan = addr.getFeUnitChannel();
	chan  = 12*( addr.getFedFeUnit() ) + addr.getFeUnitChannel();
      } catch(...) { 
	handleException( __func__, "Problem using Fed9UAddress" ); 
      } 
      
      // Determine whether FED key is inferred from cabling or channel loop
      uint32_t fed_key = 0;
      if ( summary.runType() == sistrip::FED_CABLING ) {
	fed_key = ( ( *ifed & sistrip::invalid_ ) << 16 ) | ( chan & sistrip::invalid_ );
      } else { 
	fed_key = ( ( iconn->fedId() & sistrip::invalid_ ) << 16 ) | ( iconn->fedCh() & sistrip::invalid_ );
      }
      
      // Determine whether DetId or FED key should be used to index digi containers
      uint32_t key = ( useFedKey_ || mode == sistrip::FED_SCOPE_MODE ) ? fed_key : iconn->detId();
      
      // Determine APV std::pair number (needed only when using DetId)
      uint16_t ipair = ( useFedKey_ || mode == sistrip::FED_SCOPE_MODE ) ? 0 : iconn->apvPairNumber();

      if ( mode == sistrip::FED_SCOPE_MODE ) {
	
	std::vector<uint16_t> samples; 
	try { 
   	  samples = fedEvent_->feUnit( iunit ).channel( ichan ).getSamples();
	} catch(...) { 
	  std::stringstream sss;
	  sss << "Problem accessing SCOPE_MODE data for FedId/FeUnit/FeChan: " 
	      << *ifed << "/" << iunit << "/" << ichan;
	  handleException( __func__, sss.str() ); 
	} 
	
	if ( !samples.empty() ) { 
          DetSet_SiStripDig_registry regItem(key, 0, scope_work_digis_.size(), samples.size());
          for ( uint16_t i = 0, n = samples.size(); i < n; i++ ) {
            scope_work_digis_.push_back(  SiStripRawDigi( samples[i] ) );
          }
          scope_work_registry_.push_back( regItem );
	}
	
      } else if ( mode == sistrip::FED_VIRGIN_RAW ) {

	std::vector<uint16_t> samples; 
	try {
   	  samples = fedEvent_->channel( iunit, ichan ).getSamples();
	} catch(...) { 
	  std::stringstream sss;
	  sss << "Problem accessing VIRGIN_RAW data for FED id/ch: " 
	      << *ifed << "/" << ichan;
	  handleException( __func__, sss.str() ); 
	} 
	
	if ( !samples.empty() ) { 
          DetSet_SiStripDig_registry regItem(key, 256*ipair, virgin_work_digis_.size(), samples.size());
	  uint16_t physical;
	  uint16_t readout; 
          for ( uint16_t i = 0, n = samples.size(); i < n; i++ ) {
	    physical = i%128;
	    readoutOrder( physical, readout );                 // convert index from physical to readout order
	    (i/128) ? readout=readout*2+1 : readout=readout*2; // un-multiplex data
            virgin_work_digis_.push_back(  SiStripRawDigi( samples[readout] ) );
	  }
          virgin_work_registry_.push_back( regItem );
	}

      } else if ( mode == sistrip::FED_PROC_RAW ) {

	std::vector<uint16_t> samples; 
	try {
   	  samples = fedEvent_->channel( iunit, ichan ).getSamples();
	} catch(...) { 
	  std::stringstream sss;
	  sss << "Problem accessing PROC_RAW data for FED id/ch: " 
	      << *ifed << "/" << ichan;
	  handleException( __func__, sss.str() ); 
	} 

	if ( !samples.empty() ) { 
          DetSet_SiStripDig_registry regItem(key, 256*ipair, proc_work_digis_.size(), samples.size());
          for ( uint16_t i = 0, n = samples.size(); i < n; i++ ) {
            proc_work_digis_.push_back(  SiStripRawDigi( samples[i] ) );
          }
          proc_work_registry_.push_back( regItem );
	}

      } else if ( ( mode == sistrip::FED_ZERO_SUPPR ) || 
                  ( mode == sistrip::FED_ZERO_SUPPR_LITE ) ) { 

        DetSet_SiStripDig_registry regItem(key, 0, zs_work_digis_.size(), 0);

        int fed9uIterOffset = (mode == sistrip::FED_ZERO_SUPPR ? 7 : 2); // offset to start decoding from
	try{ 
	  bool corrupt = false;
#ifdef USE_PATCH_TO_CATCH_CORRUPT_FED_DATA
	  int16_t last_strip = -1;
	  uint16_t strips = useFedKey_ ? 256 : 256 * iconn->nApvPairs();
#endif
	  Fed9U::Fed9UEventIterator fed_iter = const_cast<Fed9U::Fed9UEventChannel&>(fedEvent_->channel( iunit, ichan )).getIterator();
	  Fed9U::Fed9UEventIterator i = fed_iter + fed9uIterOffset; 
	  while ( i.size() > 0 && !corrupt ) {
	    unsigned char first_strip = *i++; // first strip of cluster
	    unsigned char width = *i++;       // cluster width in strips 
	    uint16_t istr = 0; 
	    while ( istr < width && !corrupt ) {
	      uint16_t strip = ipair*256 + first_strip + istr;
#ifdef USE_PATCH_TO_CATCH_CORRUPT_FED_DATA
	      if ( !( strip < strips && strip > last_strip ) ) { // check for corrupt FED data
		// 		if ( edm::isDebugEnabled() ) {
		// 		  std::stringstream ss;
		// 		  ss << "[SiStripRawToDigiUnpacker::" << __func__ << "]"
		// 		     << " Corrupt FED data found for FED id " << *ifed
		// 		     << " and channel " << iconn->fedCh()
		// 		     << "!  present strip: " << strip
		// 		     << "  last strip: " << last_strip
		// 		     << "  detector strips: " << strips;
		// 		  edm::LogWarning(mlRawToDigi_) << ss.str();
		// 		}
		corrupt = true; 
		break;
	      } 
	      last_strip = strip;
#endif
	      zs_work_digis_.push_back( SiStripDigi( strip, static_cast<uint16_t>(*i) ) );
	      *i++; // Iterate to next sample
	      istr++;
	    }
	  }
	} catch(...) { 
	  std::stringstream sss;
	  sss << "Problem accessing ZERO_SUPPR data for FED id/ch: " 
	      << *ifed << "/" << ichan;
	  handleException( __func__, sss.str() ); 
	} 

        regItem.length = zs_work_digis_.size() - regItem.index;
        if (regItem.length > 0) {
            regItem.first = zs_work_digis_[regItem.index].strip();
            zs_work_registry_.push_back(regItem);
        }
      } else { // Unknown readout mode! => assume scope mode
	
	if ( edm::isDebugEnabled() ) {
	  std::stringstream ss;
	  ss << "[SiStripRawToDigiUnpacker::" << __func__ << "]"
	     << " Unknown FED readout mode (" << mode
	     << ")! Assuming SCOPE MODE..."; 
	  edm::LogWarning(mlRawToDigi_) << ss.str();
	}
	
	std::vector<uint16_t> samples; 
	try {
   	  samples = fedEvent_->feUnit( iunit ).channel( ichan ).getSamples();
	} catch(...) { 
	  std::stringstream sss;
	  sss << "Problem accessing data (UNKNOWN FED READOUT MODE) for FED id/ch: " 
	      << *ifed << "/" << ichan;
	  handleException( __func__, sss.str() ); 
	} 

	if ( !samples.empty() ) { 
          DetSet_SiStripDig_registry regItem(key, 0, scope_work_digis_.size(), samples.size());
	  for ( uint16_t i = 0, n= samples.size(); i < n; i++ ) {
            scope_work_digis_.push_back(  SiStripRawDigi( samples[i] ) );
	  }
          scope_work_registry_.push_back( regItem );

	  if ( edm::isDebugEnabled() ) {
	    std::stringstream ss;
	    ss << "Extracted " << samples.size() 
	       << " SCOPE MODE digis (samples[0] = " << samples[0] 
	       << ") from FED id/ch " 
	       << iconn->fedId() << "/" << iconn->fedCh();
	    //LogTrace(mlRawToDigi_) << ss.str();
	  }
	} else {
	  if ( edm::isDebugEnabled() ) {
	    edm::LogWarning(mlRawToDigi_)
	      << "[SiStripRawToDigiUnpacker::" << __func__ << "]"
	      << " No SM digis found!"; 
	  }
	}

      }

    } // channel loop
  } // fed loop

  if ( ! zs_work_registry_.empty() ) {
        std::sort( zs_work_registry_.begin(), zs_work_registry_.end() );
        std::vector< edm::DetSet<SiStripDigi> > sorted_and_merged;
        sorted_and_merged.reserve(  std::min(zs_work_registry_.size(), size_t(17000)) );

        bool errorInData = false;
        std::vector<DetSet_SiStripDig_registry>::iterator it = zs_work_registry_.begin(), it2 = it+1, end = zs_work_registry_.end();
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
            throw cms::Exception("Bug Found") << "Container must be already sorted!\nat " << __FILE__ << ", line " << __LINE__ <<"\n";
        }
        std::vector< edm::DetSet<SiStripDigi> >::iterator iii = sorted_and_merged.begin();
        std::vector< edm::DetSet<SiStripDigi> >::iterator jjj = sorted_and_merged.end(); 
        for ( ; iii != jjj; ++iii ) { 
            if ( ! __gnu_cxx::is_sorted( iii->begin(), iii->end() ) ) {
                // this might be an error in the data, if the raws from one FED are not sorted
                iii->clear(); errorInData = true;
            }
        }
        if (errorInData) { edm::LogWarning("CorruptData") << "Some modules contained corrupted ZS raw data, and have been skipped in unpacking\n"; }
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
      std::vector<DetSet_SiStripDig_registry>::iterator it = virgin_work_registry_.begin(), it2, end = virgin_work_registry_.end();
      while (it < end) {
          sorted_and_merged.push_back( edm::DetSet<SiStripRawDigi>(it->detid) );
          std::vector<SiStripRawDigi> & digis = sorted_and_merged.back().data;

          bool isDetOk = true; 
          // first count how many digis we have
          int maxFirstStrip = it->first;
          for (it2 = it+1; (it2 != end) && (it2->detid == it->detid); ++it2) { 
              if (it2->first <= maxFirstStrip) { isDetOk = false; continue; } // duplicated APV or data corruption. DO NOT 'break' here!
              maxFirstStrip = it2->first;                           
          }
          if (!isDetOk) { errorInData = true; it = it2; continue; } // skip whole det
          // make room for 256 * (max_apv_pair + 1) Raw Digis
          digis.resize(maxFirstStrip + 256);
          // push them in
          for (it2 = it+0; (it2 != end) && (it2->detid == it->detid); ++it2) {
              if (it->length != 256)  { isDetOk = false; continue; } // data corruption. DO NOT 'break' here
              std::copy( & virgin_work_digis_[it2->index], & virgin_work_digis_[it2->index + it2->length], & digis[it2->first] );
          }
          if (!isDetOk) { errorInData = true; digis.clear(); it = it2; continue; } // skip whole det
          it = it2;
      }
      if (errorInData) { edm::LogWarning("CorruptData") << "Some modules contained corrupted virgin raw data, and have been skipped in unpacking\n"; }
      // check sorting
      if ( !__gnu_cxx::is_sorted( sorted_and_merged.begin(), sorted_and_merged.end()  ) ) {
          // this is an error in the code: i DID sort it already!
          throw cms::Exception("Bug Found") << "Container must be already sorted!\nat " << __FILE__ << ", line " << __LINE__ <<"\n";
      }
      // make output DetSetVector
      edm::DetSetVector<SiStripRawDigi> virgin_raw_dsv( sorted_and_merged, true ); 
      virgin_raw.swap( virgin_raw_dsv );
  } 
  
  // Populate final DetSetVector container with PR data 
  if ( !proc_work_registry_.empty() ) {
      std::sort( proc_work_registry_.begin(), proc_work_registry_.end() );

      std::vector< edm::DetSet<SiStripRawDigi> > sorted_and_merged;
      sorted_and_merged.reserve( std::min(proc_work_registry_.size(), size_t(17000)) );

      bool errorInData = false;
      std::vector<DetSet_SiStripDig_registry>::iterator it = proc_work_registry_.begin(), it2 = it+1, end = proc_work_registry_.end();
      while (it < end) {
          sorted_and_merged.push_back( edm::DetSet<SiStripRawDigi>(it->detid) );
          std::vector<SiStripRawDigi> & digis = sorted_and_merged.back().data;
          bool isDetOk = true; 
          // first count how many digis we have
          int maxFirstStrip = it->first;
          for (it2 = it+1; (it2 != end) && (it2->detid == it->detid); ++it2) { 
              if (it2->first <= maxFirstStrip) { isDetOk = false; continue; } // duplicated APV or data corruption. DO NOT 'break' here!
              maxFirstStrip = it2->first; 
          }
          if (!isDetOk) { errorInData = true; it = it2; continue; } // skip whole det
          // make room for 256 * (max_apv_pair + 1) Raw Digis
          digis.resize(maxFirstStrip + 256);
          // push them in
          for (it2 = it+0; (it2 != end) && (it2->detid == it->detid); ++it2) {
              if (it->length != 256)  { isDetOk = false; continue; } // data corruption. DO NOT 'break' here
              std::copy( & proc_work_digis_[it2->index], & proc_work_digis_[it2->index + it2->length], & digis[it->first] );
          }
          if (!isDetOk) { errorInData = true; digis.clear(); it = it2; continue; } // skip whole det
          it = it2;
      }
      if (errorInData) { edm::LogWarning("CorruptData") << "Some modules contained corrupted processed raw data, and have been skipped in unpacking\n"; }
      // check sorting
      if ( !__gnu_cxx::is_sorted( sorted_and_merged.begin(), sorted_and_merged.end()  ) ) {
          // this is an error in the code: i DID sort it already!
          throw cms::Exception("Bug Found") << "Container must be already sorted!\nat " << __FILE__ << ", line " << __LINE__ <<"\n";
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
      std::vector<DetSet_SiStripDig_registry>::iterator it, end;
      for (it = scope_work_registry_.begin(), end = scope_work_registry_.end() ; it != end; ++it) {
          sorted_and_merged.push_back( edm::DetSet<SiStripRawDigi>(it->detid) );
          std::vector<SiStripRawDigi> & digis = sorted_and_merged.back().data;
          digis.insert( digis.end(), & scope_work_digis_[it->index], & scope_work_digis_[it->index + it->length] );

          if ( (it +1 != end) && (it->detid == (it+1)->detid) ) {
              errorInData = true; 
              // let's skip *all* the detsets for that key, as we don't know which is the correct one!
              do { ++it; } while ( ( it+1 != end) && (it->detid == (it+1)->detid) );
              //throw cms::Exception("Duplicate Key") << "Duplicate key " << it->detid << " found in scope mode.\n"; 
          }
      }
      if (errorInData) { edm::LogWarning("CorruptData") << "Some fed keys contained corrupted scope mode data, and have been skipped in unpacking\n"; }
      // check sorting
      if ( !__gnu_cxx::is_sorted( sorted_and_merged.begin(), sorted_and_merged.end()  ) ) {
          // this is an error in the code: i DID sort it already!
          throw cms::Exception("Bug Found") << "Container must be already sorted!\nat " << __FILE__ << ", line " << __LINE__ <<"\n";
      }
      // make output DetSetVector
      edm::DetSetVector<SiStripRawDigi> scope_mode_dsv( sorted_and_merged, true ); 
      scope_mode.swap( scope_mode_dsv );

  } 

  // Increment event counter
  event_++;
  
  if ( first_ ) { first_ = false; }

  // final cleanup, just in case
  cleanupWorkVectors();
}

// -----------------------------------------------------------------------------
//
void SiStripRawToDigiUnpacker::cleanupWorkVectors() {
  // Clear working areas and registries
  zs_work_registry_.clear();      zs_work_digis_.clear();
  virgin_work_registry_.clear();  virgin_work_digis_.clear();
  proc_work_registry_.clear();    proc_work_digis_.clear();
  scope_work_registry_.clear();   scope_work_digis_.clear();
}

// -----------------------------------------------------------------------------
//
void SiStripRawToDigiUnpacker::triggerFed( const FEDRawDataCollection& buffers,
					   SiStripEventSummary& summary,
					   const uint32_t& event ) {
  
  // Pointer to data (recast as 32-bit words) and number of 32-bit words
  uint32_t* data_u32 = 0;
  uint32_t  size_u32 = 0;
  
  if ( triggerFedId_ < 0 ) { // Search mode
    
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
	    ss << "[SiStripRawToDigiUnpacker::" << __func__ << "]"
	       << " Search mode for 'trigger FED' activated!"
	       << " Found 'trigger FED' info with id " << triggerFedId_;
	    LogTrace(mlRawToDigi_) << ss.str();
	  }
	}
      }
      ifed++;
    }
    if ( triggerFedId_ < 0 ) {
      triggerFedId_ = 0;
      if ( edm::isDebugEnabled() ) {
	std::stringstream ss;
	ss << "[SiStripRawToDigiUnpacker::" << __func__ << "]"
	   << " Search mode for 'trigger FED' activated!"
	   << " 'Trigger FED' info not found!";
	edm::LogWarning(mlRawToDigi_) << ss.str();
      }
    }
    
  } else if ( triggerFedId_ > 0 ) { // "Trigger FED" id given in .cfg file
    
    const FEDRawData& trigger_fed = buffers.FEDData( triggerFedId_ );
    if ( trigger_fed.data() && trigger_fed.size() ) {
      uint8_t*  temp = const_cast<uint8_t*>( trigger_fed.data() );
      data_u32 = reinterpret_cast<uint32_t*>( temp ) + sizeof(fedh_t)/sizeof(uint32_t) + 1;
      size_u32 = trigger_fed.size()/sizeof(uint32_t) - sizeof(fedh_t)/sizeof(uint32_t) - 1;
      fedt_t* fed_trailer = reinterpret_cast<fedt_t*>( temp + trigger_fed.size() - sizeof(fedt_t) );
      if ( fed_trailer->conscheck != 0xDEADFACE ) { 
	if ( edm::isDebugEnabled() ) {
	  edm::LogWarning(mlRawToDigi_) 
	    << "[SiStripRawToDigiUnpacker::" << __func__ << "]"
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
	ss << "[SiStripRawToDigiUnpacker::" << __func__ << "]"
	   << " NULL pointer to 'trigger FED' data";
	edm::LogWarning(mlRawToDigi_) << ss.str();
      }
      return;
    } 
    if ( size_u32 < sizeof(TFHeaderDescription)/sizeof(uint32_t) ) {
      if ( edm::isDebugEnabled() ) {
	std::stringstream ss;
	ss << "[SiStripRawToDigiUnpacker::" << __func__ << "]"
	   << " Unexpected 'Trigger FED' data size [32-bit words]: " << size_u32;
	edm::LogWarning(mlRawToDigi_) << ss.str();
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
      ss << "[SiStripRawToDigiUnpacker::" << __func__ << "]"
	 << " EventSummary built from \"trigger FED\":" 
	 << std::endl << summary;
      LogTrace(mlRawToDigi_) << ss.str();
    }
    once_ = false;
  }
  
}

//------------------------------------------------------------------------------
/** 
    Removes any data appended prior to FED buffer and reorders 32-bit words if swapped.
    Pattern matches to find DAQ header:
    DAQ header,  4 MSB, BEO1, with value 0x5
    DAQ header,  4 LSB, Hx$$, with value 0x8 (or 0x0)
    DAQ trailer, 4 MSB, EOE,  with value 0xA
*/
void SiStripRawToDigiUnpacker::locateStartOfFedBuffer( const uint16_t& fed_id,
						       const FEDRawData& input,
						       FEDRawData& output ) {
  
  // Check size of input buffer
  if ( input.size() < 24 ) { 
    output.resize( input.size() ); // Return UNadjusted buffer start position and size
    memcpy( output.data(), input.data(), input.size() );
    if ( edm::isDebugEnabled() ) {
      std::stringstream ss; 
      ss << "[SiStripRawToDigiUnpacker::" << __func__ << "] "
	 << "Input FEDRawData with FED id " << fed_id 
	 << " has size " << input.size();
      edm::LogWarning(mlRawToDigi_) << ss.str();
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

    /*

      Some info on FED buffer 32-bit word swapping. Table below
      indicates if data are swapped relative to "old" VME format (as
      orignally expected by Fed9UEvent)

      | SWAPPED?    |         DATA FORMAT       |
      | (wrt "OLD") | OLD (0xED)  | NEW (0xC5)  |
      |             | VME | SLINK | VME | SLINK |
      -------------------------------------------
      | DAQ HEADER  |  N  |   Y   |  Y  |   Y   |
      | TRK HEADER  |  N  |   Y   |  N  |   N   |
      | PAYLOAD     |  N  |   Y   |  N  |   N   |
      | DAQ TRAILER |  N  |   Y   |  Y  |   Y   |

      So, in code, we check in code order of bytes in DAQ header/trailer only:
      -> if "old_vme_header",           then old format read out via vme, so do nothing.
      -> else if "old_slink_header",    then data mapy be wwapped, so check additionally the TRK header:
      --> if "old_slink_payload",       then old format read out via slink, so swap all data;
      --> else if "new_buffer_format",  then new format, handled internally by Fed9UEvent, so do nothing.

     */
    
    bool old_vme_header = 
      ( input_u32[0]    & 0xF0000000 ) == 0x50000000 &&
      ( fed_trailer[0]  & 0xF0000000 ) == 0xA0000000 && 
      ( (fed_trailer[0] & 0x00FFFFFF)*0x8 ) == (input.size() - offset);
    
    bool old_slink_header = 
      ( input_u32[1]    & 0xF0000000 ) == 0x50000000 &&
      ( fed_trailer[1]  & 0xF0000000 ) == 0xA0000000 &&
      ( (fed_trailer[1] & 0x00FFFFFF)*0x8 ) == (input.size() - offset);

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
	  ss << "[SiStripRawToDigiUnpacker::" << __func__ << "]" 
	     << " Buffer for FED id " << fed_id 
	     << " has been found at byte position " << offset
	     << " with a size of " << input.size()-offset << " bytes."
	     << " Adjust the configurable 'AppendedBytes' to " << offset;
	  LogTrace(mlRawToDigi_) << ss.str();
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
	    ss << "[SiStripRawToDigiUnpacker::" << __func__ << "]" 
	       << " Buffer (with MSB and LSB 32-bit words swapped) for FED id " << fed_id 
	       << " has been found at byte position " << offset
	       << " with a size of " << output.size() << " bytes."
	       << " Adjust the configurable 'AppendedBytes' to " << offset;
	    LogTrace(mlRawToDigi_) << ss.str();
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
	    ss << "[SiStripRawToDigiUnpacker::" << __func__ << "]" 
	       << " Buffer for FED id " << fed_id 
	       << " has been found at byte position " << offset
	       << " with a size of " << input.size()-offset << " bytes."
	       << " Adjust the configurable 'AppendedBytes' to " << offset;
	    LogTrace(mlRawToDigi_) << ss.str();
	  }
	}
	
      } else { headerBytes_ < 0 ? found = false : found = true; }
    } else { headerBytes_ < 0 ? found = false : found = true; }
    ichar++;
  }      
  
  // Check size of output buffer
  if ( output.size() == 0 ) { 
    
    // Did not find DAQ header after search => return UNadjusted buffer start position and size
    output.resize( input.size() ); 
    memcpy( output.data(), input.data(), input.size() );
    if ( edm::isDebugEnabled() ) {
      std::stringstream ss;
      if ( headerBytes_ < 0 ) {
	ss << "[SiStripRawToDigiUnpacker::" << __func__ << "]"
	   << " DAQ header not found within buffer for FED id: " << fed_id;
      } else {
	uint32_t* input_u32 = reinterpret_cast<uint32_t*>( const_cast<unsigned char*>( input.data() ) );
	ss << "[SiStripRawToDigiUnpacker::" << __func__ << "]"
	   << " DAQ header not found at expected location for FED id: " << fed_id << std::endl
	   << " First 64-bit word of buffer is 0x"
	   << std::hex 
	   << std::setfill('0') << std::setw(8) << input_u32[0] 
	   << std::setfill('0') << std::setw(8) << input_u32[1] 
	   << std::dec << std::endl
	   << " Adjust 'AppendedBytes' configurable to '-1' to activate 'search mode'";
      }
      edm::LogWarning(mlRawToDigi_) << ss.str();
    }
    
  } else if ( output.size() < 24 ) { // Found DAQ header after search, but too few words
    
    if ( edm::isDebugEnabled() ) {
      std::stringstream ss; 
      ss << "[SiStripRawToDigiUnpacker::" << __func__ << "]"
	 << " Unexpected buffer size! FEDRawData with FED id " << fed_id 
	 << " has size " << output.size();
      edm::LogWarning(mlRawToDigi_) << ss.str();
    }

  } 
  
}

// -----------------------------------------------------------------------------
//
void SiStripRawToDigiUnpacker::updateEventSummary( const Fed9U::Fed9UEvent* const fed, 
						   SiStripEventSummary& summary ) {
  
  // Retrieve contents of DAQ registers
  //@@ uint16_t trigger_type = sistrip::invalid_;
  uint16_t readout_mode = sistrip::invalid_;
  uint32_t daq1 = sistrip::invalid32_;
  uint32_t daq2 = sistrip::invalid32_;
  
  if ( fed ) {
    //@@ trigger_type = static_cast<uint16_t>( fed->?????() ); 
    readout_mode = static_cast<uint16_t>( fed->getEventType() ); 
    daq1 = static_cast<uint32_t>( fed->getDaqRegister() ); 
    daq2 = static_cast<uint32_t>( fed->getDaqRegisterTwo() ); 
  }
  
  // If FED DAQ registers contain info, update (and possibly overwrite) EventSummary 
  if ( daq1 != 0 && daq1 != sistrip::invalid32_ ) {
    
    summary.triggerFed( triggerFedId_ );
    summary.fedReadoutMode( readout_mode );
    summary.commissioningInfo( daq1, daq2 );
    
    if ( summary.isSet() && once_ ) {
      if ( edm::isDebugEnabled() ) {
	std::stringstream ss;
	ss << "[SiStripRawToDigiUnpacker::" << __func__ << "]"
	   << " EventSummary built from FED DAQ registers:"
	   << std::endl << summary;
	LogTrace(mlRawToDigi_) << ss.str();
      }
      once_ = false;
    }
  }
  //@@ below needed to set run type in PHYSICS 
//   } else if ( trigger_type == ??? ) { 
//     summary.runType( sistrip::PHYSICS );
//     std::stringstream ss;
//     ss << "[SiStripRawToDigiUnpacker::" << __func__ << "]"
//        << " EventSummary built from 'Event Type' field in DAQ header:"
//        << std::endl << summary;
//     LogTrace(mlRawToDigi_) << ss.str();
//     once_ = false;
//   }
  
}

//------------------------------------------------------------------------------
/** 
    Dumps raw data to stdout (NB: payload is byte-swapped,
    headers/trailer are not).
*/
void SiStripRawToDigiUnpacker::dumpRawData( uint16_t fed_id, 
					    const FEDRawData& buffer,
					    std::stringstream& ss ) {
  ss << "[SiStripRawToDigiUnpacker::" << __func__ << "]"
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
      if ( !tmp0 && !tmp1 && !tmp2 && !tmp3 &&
	   !tmp4 && !tmp5 && !tmp6 && !tmp7 ) { empty++; }
      else { 
	if ( empty ) { 
	  ss << "         [.." 
	     << std::dec << std::setfill('.') << std::setw(4) << empty 
	     << " null words....]" << std::endl; 
	  empty = 0; 
	}
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
      }
    }

  }
  ss << "[SiStripRawToDigiUnpacker::" << __func__ << "]"
     << " End of FED buffer";
}

// -----------------------------------------------------------------------------
// 
void SiStripRawToDigiUnpacker::handleException( std::string method_name,
						std::string extra_info ) { // throw (cms::Exception) {
  method_name = "SiStripRawToDigiUnpacker::" + method_name;
  try {
    throw; // rethrow caught exception to be dealt with below
  } 
  catch ( const cms::Exception& e ) { 
    //throw e; // rethrow cms::Exception to be caught by framework
  }
  catch ( const ICUtils::ICException& e ) {
    if ( edm::isDebugEnabled() ) {
      std::stringstream ss;
      ss << "[SiStripRawToDigiUnpacker::" << __func__ << "]"
	 << " Caught ICUtils::ICException exception!" << std::endl;
      if ( extra_info != "" ) { 
	ss << " Information: " << extra_info << std::endl;
      }
      ss << " Caught ICUtils::ICException in ["
	 << method_name << "] with message:" << std::endl 
	 << e.what();
      edm::LogWarning(mlRawToDigi_) << ss.str();
    }
    //throw cms::Exception(mlRawToDigi_) << ss.str();
  }
  catch ( const std::exception& e ) {
    if ( edm::isDebugEnabled() ) {
      std::stringstream ss;
      ss << "[SiStripRawToDigiUnpacker::" << __func__ << "]"
	 << " Caught std::exception!" << std::endl;
      if ( extra_info != "" ) { 
	ss << " Information: " << extra_info << std::endl;
      }
      ss << " Caught std::exception in ["
	 << method_name << "] with message:" << std::endl 
	 << e.what();
      edm::LogWarning(mlRawToDigi_) << ss.str();
    }
    //throw cms::Exception(mlRawToDigi_) << ss.str();
  }
  catch (...) {
    if ( edm::isDebugEnabled() ) {
      std::stringstream ss;
      ss << "[SiStripRawToDigiUnpacker::" << __func__ << "]"
	 << " Caught unknown exception!" << std::endl;
      if ( extra_info != "" ) { 
	ss << " Information: " << extra_info << std::endl;
      }
      ss << "Caught unknown exception in ["
	 << method_name << "]" << std::endl;
      edm::LogWarning(mlRawToDigi_) << ss.str();
    }
    //throw cms::Exception(mlRawToDigi_) << ss.str();
  }
}

