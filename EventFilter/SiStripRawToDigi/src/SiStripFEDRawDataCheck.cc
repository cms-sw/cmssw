#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDRawDataCheck.h"
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "EventFilter/SiStripRawToDigi/interface/SiStripRawToDigiUnpacker.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "interface/shared/fed_trailer.h"
#include "Fed9UUtils.hh"
#include "ICException.hh"
#include <sstream>
#include <vector>
#include <map>

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
// 
SiStripFEDRawDataCheck::SiStripFEDRawDataCheck( const edm::ParameterSet& pset ) 
  : label_( pset.getUntrackedParameter<string>("ProductLabel","source") ),
    instance_( pset.getUntrackedParameter<string>("ProductInstance","") ),
    unpacker_( new SiStripRawToDigiUnpacker(0,0,0,0,0) )
{
  if ( edm::isDebugEnabled() ) {
    LogTrace(mlRawToDigi_)
      << "[SiStripFEDRawDataCheck::" << __func__ << "]"
      << " Constructing object...";
  }
}

// -----------------------------------------------------------------------------
// 
SiStripFEDRawDataCheck::~SiStripFEDRawDataCheck() 
{
  if ( edm::isDebugEnabled() ) {
    LogTrace(mlRawToDigi_)
      << "[SiStripFEDRawDataCheck::" << __func__ << "]"
      << " Destructing object...";
  }
  if ( unpacker_ ) { delete unpacker_; }
}

// -----------------------------------------------------------------------------
// 
void SiStripFEDRawDataCheck::analyze( const edm::Event& event,
				      const edm::EventSetup& setup ) {
  
  // Retrieve FED cabling object
  edm::ESHandle<SiStripFedCabling> cabling;
  setup.get<SiStripFedCablingRcd>().get( cabling );
  
  // Retrieve FEDRawData collection
  edm::Handle<FEDRawDataCollection> buffers;
  event.getByLabel( label_, instance_, buffers ); 
  
  // Local cache of fed ids from cabling
  vector<uint16_t> fed_ids = cabling->feds();
  
  // Containers storing fed ids and buffer sizes
  typedef pair<uint16_t,uint16_t> Fed;
  typedef vector<Fed> Feds;
  Feds trg_feds; // trigger feds
  Feds trk_feds; // tracker feds
  Feds non_trk;  // feds from other sub-dets
  Feds in_data;  // feds in data but missing from cabling
  Feds in_cabl;  // feds in cabling but missing from data
  
  for ( uint16_t ifed = 0; ifed <= sistrip::CMS_FED_ID_MAX; ++ifed ) {
    const FEDRawData& fed = buffers->FEDData( static_cast<int>(ifed) );
    uint8_t* data = const_cast<uint8_t*>( fed.data() ); // look for trigger fed
    fedt_t* fed_trailer = reinterpret_cast<fedt_t*>( data + fed.size() - sizeof(fedt_t) );
    if ( fed.size() && fed_trailer->conscheck == 0xDEADFACE ) { trg_feds.push_back(Fed(ifed,fed.size())); }
    else {
      if ( ifed < sistrip::FED_ID_MIN ||
	   ifed > sistrip::FED_ID_MAX ) {
	if ( fed.size() ) { non_trk.push_back(Fed(ifed,fed.size())); }
      } else { 
	bool found = find( fed_ids.begin(), fed_ids.end(), ifed ) != fed_ids.end();
	if ( fed.size() ) {
	  if ( found ) { trk_feds.push_back(Fed(ifed,fed.size())); }
	  else { in_data.push_back(Fed(ifed,fed.size())); }
	}
	if ( found && fed.size() == 0 ) { in_cabl.push_back(Fed(ifed,0)); }
      }	
    }
  }

  // Trigger FEDs
  if ( edm::isDebugEnabled() ) {
    stringstream ss;
    ss << "[SiStripFEDRawDataCheck::" << __func__ << "]"
       << " Found " << trg_feds.size() 
       <<  " trigger FEDs";
    edm::LogVerbatim(mlRawToDigi_) << ss.str();
    ss << " with ids (and buffer sizes [bytes]): ";
    for ( uint16_t ifed = 0; ifed < trg_feds.size(); ++ifed ) {
      ss << trg_feds[ifed].first << "(" << trg_feds[ifed].second << ") ";
    }
    LogTrace(mlRawToDigi_) << ss.str();
  }

  // Not strip tracker FEDs
  if ( edm::isDebugEnabled() ) {
    stringstream ss;
    ss << "[SiStripFEDRawDataCheck::" << __func__ << "]"
       << " Found " << non_trk.size() 
       <<  " non-tracker FEDs from other sub-dets";
    edm::LogVerbatim(mlRawToDigi_) << ss.str();
    ss << " with ids (and buffer sizes [bytes]): ";
    for ( uint16_t ifed = 0; ifed < non_trk.size(); ++ifed ) {
      ss << non_trk[ifed].first << "(" << non_trk[ifed].second << ") ";
    }
    LogTrace(mlRawToDigi_) << ss.str();
  }

  // Strip tracker FEDs
  if ( edm::isDebugEnabled() ) {
    stringstream ss;
    ss << "[SiStripFEDRawDataCheck::" << __func__ << "]"
       << " Found " << trk_feds.size() 
       <<  " strip tracker FEDs";
    edm::LogVerbatim(mlRawToDigi_) << ss.str();
    ss << " with ids (and buffer sizes [bytes]): ";
    for ( uint16_t ifed = 0; ifed < trk_feds.size(); ++ifed ) {
      ss << trk_feds[ifed].first << "(" << trk_feds[ifed].second << ") ";
    }
    LogTrace(mlRawToDigi_) << ss.str();
  }
  
  // FEDs in data but missing from cabling
  if ( edm::isDebugEnabled() ) {
    stringstream ss;
    ss << "[SiStripFEDRawDataCheck::" << __func__ << "]"
       << " Found " << in_data.size() 
       <<  " FEDs in data but missing from cabling";
    edm::LogVerbatim(mlRawToDigi_) << ss.str();
    ss << " with ids (and buffer sizes [bytes]): ";
    for ( uint16_t ifed = 0; ifed < in_data.size(); ++ifed ) {
      ss << in_data[ifed].first << "(" << in_data[ifed].second << ") ";
    }
    LogTrace(mlRawToDigi_) << ss.str();
  }

  // FEDs in cabling but missing from data 
  if ( edm::isDebugEnabled() ) {
    stringstream ss;
    ss << "[SiStripFEDRawDataCheck::" << __func__ << "]"
       << " Found " << in_cabl.size() 
       <<  " FEDs in cabling but missing from data";
    edm::LogVerbatim(mlRawToDigi_) << ss.str();
    ss << " with ids (and zero buffer size): ";
    for ( uint16_t ifed = 0; ifed < in_cabl.size(); ++ifed ) {
      ss << in_cabl[ifed].first << " ";
    }
    LogTrace(mlRawToDigi_) << ss.str();
  }

  // Containers storing fed ids and channels
  typedef vector<uint16_t> Channels;
  typedef map<uint16_t,Channels> ChannelsMap;
  ChannelsMap construct; // errors on contruction of object
  ChannelsMap check;     // errors on check event
  ChannelsMap address;
  ChannelsMap channels_with_data;
  ChannelsMap channels_missing_data;
  ChannelsMap cabling_missing_channels;
  ChannelsMap connected;
  
  // Analyze strip tracker FED buffers in data
  Fed9U::Fed9UEvent* fed_event = new Fed9U::Fed9UEvent();
  for ( uint16_t ii = 0; ii < trk_feds.size(); ++ii ) {
    uint16_t ifed = trk_feds[ii].first;
    const FEDRawData& fed = buffers->FEDData( static_cast<int>(ifed) );
    
    FEDRawData output; 
    if ( unpacker_ ) { unpacker_->locateStartOfFedBuffer( ifed, fed, output ); }

    Fed9U::u32* data_u32 = reinterpret_cast<Fed9U::u32*>( const_cast<unsigned char*>( output.data() ) );
    Fed9U::u32  size_u32 = static_cast<Fed9U::u32>( output.size() / 4 ); 
    
    try {
      fed_event->Init( data_u32, 0, size_u32 ); 
    } catch ( const ICUtils::ICException& e ) {
      construct[ifed].push_back(0); //@@ error code here?
      continue;
    }
    
    try {
      fed_event->checkEvent();
    } catch ( const ICUtils::ICException& e ) {
      check[ifed].push_back(0); //@@ error code here?
      continue;
    } 

    vector<FedChannelConnection> channels = cabling->connections(ifed);
    for ( uint16_t chan = 0; chan < channels.size(); ++chan ) {
      if ( channels[chan].isConnected() ) { connected[ifed].push_back(channels[chan].fedCh()); }
    }

    for ( uint16_t iunit = 0; iunit < 8; ++iunit ) {
      for ( uint16_t ichan = 0; ichan < 12; ++ichan ) {
	Fed9U::Fed9UAddress addr;

	// record Fed9UAddresses that throw exceptions
	try {
	  addr.setAddress( static_cast<unsigned char>( iunit ),
			   static_cast<unsigned char>( ichan ) );
	} catch ( const ICUtils::ICException& e ) {
	  address[ifed].push_back(addr.getFedChannel());
	  continue;
	} 

  	try { 
   	  fed_event->channel( addr );
	} catch ( const ICUtils::ICException& e ) {
	  // record channels in cabling but without data
	  bool found = find( connected[ifed].begin(), 
			     connected[ifed].end(), 
			     addr.getFedChannel() ) != connected[ifed].end();
	  if ( found ) { channels_missing_data[ifed].push_back(addr.getFedChannel()); }
	  continue;
	} 
	
	// record channels with data
	channels_with_data[ifed].push_back(addr.getFedChannel());

      }
    }

    // check if channels with data are in cabling
    uint16_t chan = 0; 
    bool found = true;
    while ( chan < connected[ifed].size() && found ) {
      Channels::iterator iter = channels_with_data[ifed].begin();
      while ( iter < channels_with_data[ifed].end() ) {
	if ( *iter == connected[ifed][chan] ) { break; }
	iter++;
      }
      found = iter != channels_with_data[ifed].end();
      if ( !found ) { 
	cabling_missing_channels[ifed].push_back(connected[ifed][chan]); 
	//channels_with_data[ifed].erase(iter);
      }
      chan++;
    }

  } // fed loop
  if ( fed_event ) { delete fed_event; }


  // constructing fed9uevents
  if ( edm::isDebugEnabled() ) {
    stringstream ss;
    ss << "[SiStripFEDRawDataCheck::" << __func__ << "]"
       << " Found " << construct.size() 
       <<  " FED buffers that fail on construction of a Fed9UEvent";
    edm::LogVerbatim(mlRawToDigi_) << ss.str();
    ss << " with ids: ";
    for ( ChannelsMap::iterator iter = construct.begin(); iter != construct.end(); ++iter ) {
      ss << iter->first << " ";
    }
    LogTrace(mlRawToDigi_) << ss.str();
  }

  // "check event" on fed9uevents
  if ( edm::isDebugEnabled() ) {
    stringstream ss;
    ss << "[SiStripFEDRawDataCheck::" << __func__ << "]"
       << " Found " << check.size() 
       <<  " FED buffers that fail on Fed9UEvent::checkEvent()";
    edm::LogVerbatim(mlRawToDigi_) << ss.str();
    ss << " with ids: ";
    for ( ChannelsMap::iterator iter = check.begin(); iter != check.end(); ++iter ) {
      ss << iter->first << " ";
    }
    LogTrace(mlRawToDigi_) << ss.str();
  }

  // fed9uaddress exceptions
  if ( edm::isDebugEnabled() ) {
    stringstream ss;
    ss << "[SiStripFEDRawDataCheck::" << __func__ << "]"
       << " Found " << address.size() 
       <<  " FED buffers that throw Fed9UAddress exceptions";
    edm::LogVerbatim(mlRawToDigi_) << ss.str();
    ss << " with ids (and number of exceptions): ";
    for ( ChannelsMap::iterator iter = address.begin(); iter != address.end(); ++iter ) {
      ss << iter->first << "(" 
	 << iter->second.size() << ") ";
    }
    LogTrace(mlRawToDigi_) << ss.str();
  }

  //  connected channels with data
  if ( edm::isDebugEnabled() ) {
    stringstream ss;
    ss << "[SiStripFEDRawDataCheck::" << __func__ << "]"
       << " Found " << channels_with_data.size() 
       <<  " FED buffers with data in connected channels";
    edm::LogVerbatim(mlRawToDigi_) << ss.str();
    ss << " with ids (and number of connected channels with data): ";
    for ( ChannelsMap::iterator iter = channels_with_data.begin(); iter != channels_with_data.end(); ++iter ) {
      ss << iter->first << "(" 
	 << iter->second.size() << ") ";
    }
    LogTrace(mlRawToDigi_) << ss.str();
  }

  //  connected channels with missing data
  if ( edm::isDebugEnabled() ) {
    stringstream ss;
    ss << "[SiStripFEDRawDataCheck::" << __func__ << "]"
       << " Found " << channels_missing_data.size() 
       <<  " FED buffers with connected channels and missing data";
    edm::LogVerbatim(mlRawToDigi_) << ss.str();
    ss << " with ids (and number of connected channels with missing data): ";
    for ( ChannelsMap::iterator iter = channels_missing_data.begin(); iter != channels_missing_data.end(); ++iter ) {
      ss << iter->first << "(" 
	 << iter->second.size() << ") ";
    }
    LogTrace(mlRawToDigi_) << ss.str();
  }

  //  channels with data but not in cabling
  if ( edm::isDebugEnabled() ) {
    stringstream ss;
    ss << "[SiStripFEDRawDataCheck::" << __func__ << "]"
       << " Found " << cabling_missing_channels.size() 
       <<  " FED buffers with data in channels that are missing from cabling";
    edm::LogVerbatim(mlRawToDigi_) << ss.str();
    ss << " with ids (number of channels containing data that are missing from cabling, channel number): ";
    for ( ChannelsMap::iterator iter = cabling_missing_channels.begin(); iter != cabling_missing_channels.end(); ++iter ) {
      ss << iter->first << "(" 
	 << iter->second.size() << ":";
      for ( Channels::iterator jter = iter->second.begin(); jter != iter->second.end(); ++jter ) {
	ss << *jter << ",";
      }
      ss << ") ";
    }
    LogTrace(mlRawToDigi_) << ss.str();
  }

}
