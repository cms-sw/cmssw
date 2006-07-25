#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>
#include <string>

using namespace std;

// -----------------------------------------------------------------------------
//
SiStripFedCabling::SiStripFedCabling( const vector<FedChannelConnection>& input ) 
  : feds_(),
    connected_(),
    detected_(),
    undetected_()
{
  edm::LogInfo("FedCabling") << "[SiStripFedCabling::SiStripFedCabling] Constructing object...";
  buildFedCabling( input );
}

// -----------------------------------------------------------------------------
//
SiStripFedCabling::SiStripFedCabling() 
  : feds_(),
    connected_(),
    detected_(),
    undetected_()
{
  edm::LogInfo("FedCabling") << "[SiStripFedCabling::SiStripFedCabling] Constructing object...";
}

// -----------------------------------------------------------------------------
//
SiStripFedCabling::~SiStripFedCabling() {
  edm::LogInfo("FedCabling") << "[SiStripFedCabling::~SiStripFedCabling] Destructing object...";
}

// -----------------------------------------------------------------------------
//
void SiStripFedCabling::buildFedCabling( const vector<FedChannelConnection>& input ) {
  static const string method = "SiStripFedCabling::buildFedCabling";
  edm::LogVerbatim("FedCabling") << "["<<method<<"] Building FED cabling...";
  
  // Check input
  if ( input.empty() ) {
    edm::LogError("FedCabling") << "["<<method<<"] Input vector of zero size!"; 
  }
  
  static const uint16_t MaxFedId = 1024;
  static const uint16_t MaxFedCh = 96;
  
  // Clear containers
  connected_.clear(); 
  detected_.clear();
  undetected_.clear();
  
  // Iterate through FEDs
  for ( uint16_t iconn = 0; iconn < input.size(); iconn++ ) {
    
    uint16_t fed_id = input[iconn].fedId();
    uint16_t fed_ch = input[iconn].fedCh();

    // Check on FED ids and channels
    if ( fed_id >= MaxFedId ) {
      edm::LogError("FedCabling") << "["<<method<<"] Unexpected FED id! " << fed_id; 
    } 
    if ( fed_ch >= MaxFedCh ) {
      edm::LogError("FedCabling") << "["<<method<<"] Unexpected FED channel! " << fed_ch;
    } 
    
    // Resize container to accommodate all FED channels
    if ( connected_.size() <= fed_id ) { connected_.resize(fed_id+1); }
    if ( connected_[fed_id].size() != 96 ) { connected_[fed_id].resize(96); }
    
    // Fill appropriate container
    bool detected  = true; //@@ input[iconn].i2cAddr0() || input[iconn].i2cAddr1();
    bool connected = input[iconn].fedId();
    if ( detected && connected ) {
      connected_[fed_id][fed_ch] = input[iconn];
      vector<uint16_t>::iterator id = find( feds_.begin(), feds_.end(), fed_id );
      if ( id == feds_.end() ) { feds_.push_back( fed_id ); }
    }
    //       } else if ( detected && !connected ) {
    // 	detected_.push_back( input[iconn] );
    //       } else if ( !detected && !connected ) {
    // 	undetected_.push_back( input[iconn] );
    //       }
    
  }
  
  edm::LogVerbatim("Cabling") << "["<<method<<"] Printing FedChannelConnections: ";
  vector<uint16_t>::const_iterator ifed;
  for ( ifed = feds().begin(); ifed != feds().end(); ifed++ ) {
    uint16_t connected = 0;
    vector<FedChannelConnection>::const_iterator ichan = connections(*ifed).begin();
    for ( ; ichan != connections(*ifed).end(); ichan++ ) { 
      if ( ichan->fedId() ) { 
	stringstream ss;
	ichan->print(ss); 
	LogTrace("FedCabling") << "["<<method<<"]" << ss.str();
	connected++; 
      }
    }
    edm::LogVerbatim("FedCabling") << "["<<method<<"]"
				   << " Found FED with id " << *ifed
				   << " that has " << connected
				   << " connected channels";
  }
  
}

// -----------------------------------------------------------------------------
// Returns active FEDs
const vector<uint16_t>& SiStripFedCabling::feds() const {
  return feds_;
}

// -----------------------------------------------------------------------------
// Returns connection info for FE devices connected to given FED id and channel
const FedChannelConnection& SiStripFedCabling::connection( uint16_t fed_id, 
							   uint16_t fed_chan ) const {
  if ( !connected_.empty() ) {
    if ( fed_id < connected_.size() ) {
      if ( !connected_[fed_id].empty() ) {
	if ( fed_chan < connected_[fed_id].size() ) {
	  return connected_[fed_id][fed_chan];
	} else {
	  edm::LogError("FedCabling") << "[SiStripFedCabling::connection]" 
				      << " FED channel (" << fed_chan
				      << ") is greater than or equal to vector size (" 
				      << connected_[fed_chan].size() << ")!";
	}
      } else {
	edm::LogError("FedCabling") << "[SiStripFedCabling::connection]" 
				    << " Cabling map is empty for FED id "
				    << fed_id;
      }
    } else {
      edm::LogError("FedCabling") << "[SiStripFedCabling::connection]" 
				  << " FED id (" << fed_id
				  << ") is greater than or equal to vector size (" 
				  << connected_.size() << ")!";
    }
  } else {
    edm::LogError("FedCabling") << "[SiStripFedCabling::connection]" 
				<< " Cabling map is empty!";
  }
  
  static FedChannelConnection connection; 
  return connection;
  
}

// -----------------------------------------------------------------------------
// Returns connection info for FE devices connected to given FED 
const vector<FedChannelConnection>& SiStripFedCabling::connections( uint16_t fed_id ) const {
  
  if ( !connected_.empty() ) {
    if ( fed_id < connected_.size() ) {
      if ( !connected_[fed_id].empty() ) {
	return connected_[fed_id];
      } else {
	edm::LogError("FedCabling") << "[SiStripFedCabling::connections]" 
				    << " Cabling map is empty for FED id "
				    << fed_id;
      }
    } else {
      edm::LogError("FedCabling") << "[SiStripFedCabling::connections]" 
				  << " FED id (" << fed_id
				  << ") is greater than or equal to vector size (" 
				  << connected_.size() << ")!";
    }
  } else {
    edm::LogError("FedCabling") << "[SiStripFedCabling::connections]" 
				<< " Cabling map is empty!";
  }
  
  static vector<FedChannelConnection> connections; 
  return connections;
  
}

