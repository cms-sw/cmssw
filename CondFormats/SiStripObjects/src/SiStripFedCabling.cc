#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include <iostream>
#include <string>

// -----------------------------------------------------------------------------
//
SiStripFedCabling::SiStripFedCabling( const std::vector<FedChannelConnection>& input ) :
  feds_(),
  connected_(),
  detected_(),
  undetected_()
{
  std::cout << "[SiStripFedCabling::SiStripFedCabling]" 
	    << " Constructing object..." << std::endl;

  // Check input
  if ( input.empty() ) {
    std::cerr << "[SiStripFedCabling::SiStripFedCabling]"
	      << " Input vector of zero size! " << std::endl; 
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
      std::cerr << "[SiStripFedCabling::SiStripFedCabling]"
		<< " Unexpected FED id! " << fed_id << std::endl; 
    } 
    if ( fed_ch >= MaxFedCh ) {
      std::cerr << "[SiStripFedCabling::SiStripFedCabling]"
		<< " Unexpected FED channel! " << fed_ch << std::endl;
    } 
    
    // Resize container to accommodate all FED channels
    if ( connected_.size() <= fed_id ) { connected_.resize(fed_id+1); }
    if ( connected_[fed_id].size() != 96 ) { connected_[fed_id].resize(96); }
    
    // Fill appropriate container
    bool detected  = 1; //@@ input[iconn].i2cAddr0() || input[iconn].i2cAddr1();
    bool connected = input[iconn].fedId();
    if ( detected && connected ) {
      connected_[fed_id][fed_ch] = input[iconn];
      std::vector<uint16_t>::iterator id = find( feds_.begin(), feds_.end(), fed_id );
      if ( id == feds_.end() ) { feds_.push_back( fed_id ); }
    }
    //       } else if ( detected && !connected ) {
    // 	detected_.push_back( input[iconn] );
    //       } else if ( !detected && !connected ) {
    // 	undetected_.push_back( input[iconn] );
    //       }
    
  }

  std::vector<uint16_t>::const_iterator ifed;
  for ( ifed = (*this).feds().begin(); ifed != (*this).feds().end(); ifed++ ) {
    std::cout << "[SiStripFedCabling::SiStripFedCabling]"
	      << " Found FED with id " << *ifed
	      << " that has " << (*this).connections( *ifed ).size()
	      << " connected channels" << std::endl;
  }
  
}

// -----------------------------------------------------------------------------
//
SiStripFedCabling::~SiStripFedCabling() {
  std::cout << "[SiStripFedCabling::~SiStripFedCabling]"
	    << " Destructing object..." << std::endl;
}

// -----------------------------------------------------------------------------
// Returns active FEDs
const std::vector<uint16_t>& SiStripFedCabling::feds() const {
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
	  std::cerr << "[SiStripFedCabling::connection]" 
		    << " FED channel (" << fed_chan
		    << ") is greater than or equal to vector size (" 
		    << connected_[fed_chan].size() << ")!" << std::endl;
	}
      } else {
	std::cerr << "[SiStripFedCabling::connection]" 
		  << " Cabling map is empty for FED id "
		  << fed_id << std::endl;
      }
    } else {
      std::cerr << "[SiStripFedCabling::connection]" 
		<< " FED id (" << fed_id
		<< ") is greater than or equal to vector size (" 
		<< connected_.size() << ")!" << std::endl;
    }
  } else {
    std::cerr << "[SiStripFedCabling::connection]" 
	      << " Cabling map is empty!" << std::endl;
  }
  
  static FedChannelConnection connection; 
  return connection;
  
}

// -----------------------------------------------------------------------------
// Returns connection info for FE devices connected to given FED 
const std::vector<FedChannelConnection>& SiStripFedCabling::connections( uint16_t fed_id ) const {
  
  if ( !connected_.empty() ) {
    if ( fed_id < connected_.size() ) {
      if ( !connected_[fed_id].empty() ) {
	return connected_[fed_id];
      } else {
	std::cerr << "[SiStripFedCabling::connections]" 
		  << " Cabling map is empty for FED id "
		  << fed_id << std::endl;
      }
    } else {
      std::cerr << "[SiStripFedCabling::connections]" 
		<< " FED id (" << fed_id
		<< ") is greater than or equal to vector size (" 
		<< connected_.size() << ")!" << std::endl;
    }
  } else {
    std::cerr << "[SiStripFedCabling::connections]" 
	      << " Cabling map is empty!" << std::endl;
  }
  
  static std::vector<FedChannelConnection> connections; 
  return connections;
  
}

