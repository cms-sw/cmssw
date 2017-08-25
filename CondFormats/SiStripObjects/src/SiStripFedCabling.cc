#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <iomanip>

using namespace sistrip;


// -----------------------------------------------------------------------------
//
SiStripFedCabling::SiStripFedCabling( ConnsConstIterRange input ) 
  : feds_(),
    registry_(),
    connections_(),
    detected_(),
    undetected_()
{
  LogTrace(mlCabling_)
    << "[SiStripFedCabling::" << __func__ << "]"
    << " Constructing object from connection range...";
  buildFedCabling( input );
}

// -----------------------------------------------------------------------------
//
SiStripFedCabling::SiStripFedCabling( const SiStripFedCabling& input ) 
  : feds_( input.feds_ ),
    registry_( input.registry_ ),
    connections_( input.connections_ ),
    detected_( input.detected_ ),
    undetected_( input.undetected_ )
{
  LogTrace(mlCabling_)
    << "[SiStripFedCabling::" << __func__ << "]"
    << " Copy constructing object...";
}

// -----------------------------------------------------------------------------
//
SiStripFedCabling::SiStripFedCabling() 
  : feds_(),
    registry_(),
    connections_(),
    detected_(),
    undetected_()
{
  LogTrace(mlCabling_) 
    << "[SiStripFedCabling::" << __func__ << "]"
    << " Default constructing object...";
}

// -----------------------------------------------------------------------------
//
SiStripFedCabling::~SiStripFedCabling() {
  LogTrace(mlCabling_)
    << "[SiStripFedCabling::" << __func__ << "]"
    << " Destructing object...";
}

// -----------------------------------------------------------------------------
//
void SiStripFedCabling::buildFedCabling( ConnsConstIterRange input ) {
  
  // Check input
  if ( input.empty() ) {
    edm::LogError(mlCabling_)
      << "[SiStripFedCabling::" << __func__ << "]"
      << " Input vector of FedChannelConnections is of zero size!"
      << " Unable to populate FED cabling object!"; 
    return;
  }
  
  std::stringstream ss;
  ss << "[SiStripFedCabling::" << __func__ << "]"
     << " Building FED cabling from " 
     << input.size()
     << " connections...";
  LogTrace(mlCabling_) << ss.str();
  
  // Sort input vector by FED id and channel
  Conns temp( input.size() );
  std::copy( input.begin(), input.end(), temp.begin() );
  std::sort( temp.begin(), temp.end() );
  
  // Strip FED ids
  uint16_t min_id = static_cast<uint16_t>( FEDNumbering::MINSiStripFEDID );
  uint16_t max_id = static_cast<uint16_t>( FEDNumbering::MAXSiStripFEDID );
  uint16_t nfeds  = max_id - min_id + 1;
  
  // Initialise containers
  connections_.clear();
  connections_.reserve( 96 * nfeds );
  registry_.clear();
  feds_.clear();
  registry_.resize( nfeds, ConnsRange::emptyPair() );
  
  // Populate container
  ConnsIter ii = temp.begin(); 
  ConnsIter jj = temp.end(); 
  for ( ; ii != jj; ++ii ) {
    
    uint16_t fed_id = ii->fedId();
    uint16_t fed_ch = ii->fedCh();
    uint16_t index  = fed_id - min_id;
    
    if ( fed_id < min_id || fed_id > max_id ) { continue; }
    if ( index >= registry_.size() ) { continue; }
    if ( !ii->isConnected() ) { continue; }
    
    FedsConstIter iter = find( feds_.begin(), feds_.end(), fed_id );
    if ( iter == feds_.end() ) { feds_.push_back( fed_id ); }
    
    if ( registry_[index] == ConnsRange::emptyPair() ) {
      ConnsPair conns_pair;
      conns_pair.first = std::distance( connections_.begin(), connections_.end() );
      connections_.insert( connections_.end(), 96, FedChannelConnection() ); 
      conns_pair.second = std::distance( connections_.begin(), connections_.end() );
      registry_[index] = conns_pair;
    } 

    ConnsRange conns = range( registry_[index] );
    ConnsConstIter iconn = conns.begin() + fed_ch;
    FedChannelConnection& conn = const_cast<FedChannelConnection&>(*iconn);
    conn = *ii;

  }
  
}

// -----------------------------------------------------------------------------
//
SiStripFedCabling::ConnsRange::ConnsRange( const Conns& c, ConnsPair p ) :
  vector_( c.begin(), c.end() ),
  range_( c.begin()+p.first, c.begin()+p.second )
{
  if ( p.first > p.second ||
       p.first == sistrip::invalid32_ ||
       p.second == sistrip::invalid32_ ||
       p.first > c.size() || 
       p.second > c.size() ) {
    range_ = ConnsConstIterRange( c.end(), c.end() );
  }
}

// -----------------------------------------------------------------------------
//
SiStripFedCabling::ConnsRange::ConnsRange( const Conns& c ) :
  vector_( c.begin(), c.end() ),
  range_( c.end(), c.end() )
{;}  

// -----------------------------------------------------------------------------
//
void SiStripFedCabling::ConnsRange::print( std::stringstream& ss ) const {
  ss << "[SiStripFedCabling::ConnsRange::" << __func__ << "] Debug info:" << std::endl
     << " Vector  : " << std::endl
     << "  size   : " << vector_.size() << std::endl
     << "  begin  : " 
     << std::hex << std::setfill('0') << std::setw(8)
     << &*vector_.begin()
     << std::dec << std::endl
     << "  end    : " 
     << std::hex << std::setfill('0') << std::setw(8)
     << &*vector_.end()
     << std::dec << std::endl
     << " Range   : " << std::endl
     << "  size   : "  << range_.size() << std::endl
     << "  begin  : " 
     << std::hex << std::setfill('0') << std::setw(8)
     << &*range_.begin() 
     << std::dec
     << " (dist=" << std::distance( vector_.begin(), range_.begin() ) << ")" 
     << std::endl
     << "  end    : " 
     << std::hex << std::setfill('0') << std::setw(8)
     << &*range_.end() 
     << std::dec
     << " (dist=" << std::distance( vector_.begin(), range_.end() ) << ")" 
     << std::endl
     << " Offsets : " << std::endl
     << "  first  : " << connsPair().first << std::endl
     << "  second : " << connsPair().second << std::endl;
}

// -----------------------------------------------------------------------------
//
std::ostream& operator<<( std::ostream& os, const SiStripFedCabling::ConnsRange& input ) {
  std::stringstream ss;
  input.print(ss);
  os << ss.str();
  return os;
}

// -----------------------------------------------------------------------------
// Returns connection info for FE devices connected to given FED 
SiStripFedCabling::ConnsConstIterRange SiStripFedCabling::fedConnections( uint16_t fed_id ) const {
  uint16_t index = fed_id - FEDNumbering::MINSiStripFEDID;
  if ( index < registry_.size() ) { 
    return range( registry_[ index ] ).range();
  } else { return range( registry_[ index ] ).invalid(); }
}

// -----------------------------------------------------------------------------
// Returns connection info for FE devices connected to given FED id and channel
FedChannelConnection SiStripFedCabling::fedConnection( uint16_t fed_id, 
						       uint16_t fed_ch ) const {
  ConnsConstIterRange conns = fedConnections( fed_id );
  if ( !conns.empty() && conns.size() == 96 && fed_ch < 96 ) {
    return *( conns.begin() + fed_ch ); 
  } else { return FedChannelConnection(); }
}

// -----------------------------------------------------------------------------
// 
void SiStripFedCabling::printDebug( std::stringstream& ss, const TrackerTopology* /*trackerTopo*/ ) const {

  uint16_t total = 0;
  uint16_t nfeds = 0;
  uint16_t cntr = 0;

  if ( feds_.empty() ) {
    ss << "[SiStripFedCabling::" << __func__ << "]"
       << " No FEDs found! Unable to  print cabling map!";
    return;
  } else {
    ss << "[SiStripFedCabling::" << __func__ << "]"
       << " Printing cabling map for " << feds_.size()
       << " FEDs with following ids: ";
  }
  
  std::vector<uint16_t>::const_iterator ii = feds_.begin(); 
  std::vector<uint16_t>::const_iterator jj = feds_.end(); 
  for ( ; ii != jj; ++ii ) { ss << *ii << " "; }
  ss << std::endl << std::endl;
  
  std::vector<uint16_t>::const_iterator ifed = feds_.begin(); 
  std::vector<uint16_t>::const_iterator jfed = feds_.end(); 
  for ( ; ifed != jfed; ++ifed ) {
    
    uint16_t index = *ifed - FEDNumbering::MINSiStripFEDID;
    if ( index < registry_.size() ) { 
      ConnsRange conns = range( registry_[ index ] );
      
      ss << " Printing cabling information for FED id " << *ifed 
	 << " (found " << conns.size() 
	 << " FedChannelConnection objects...)"
	 << std::endl;
      
      uint16_t ichan = 0;
      uint16_t connected = 0;
      ConnsConstIter iconn = conns.begin();
      ConnsConstIter jconn = conns.end();
      for ( ; iconn != jconn; ++iconn ) { 
	if ( iconn->fedId() != sistrip::invalid_ ) { 
	  connected++; 
	  ss << *iconn << std::endl;
	} else {
	  ss << "  (FedId/Ch " << *ifed << "/" << ichan 
	     << ": unconnected channel...)" << std::endl;
	  cntr++;
	}
	ichan++;
      } 
      
      ss << " Found " << connected 
	 << " connected channels for FED id " << *ifed << std::endl
	 << std::endl;
      if ( connected ) { nfeds++; total += connected; }
      
    }
    
  }
  
  float percent = (100.*cntr) / (96.*nfeds);
  percent = static_cast<uint16_t>( 10.*percent );
  percent /= 10.;
  ss << " Found " << total 
     << " APV pairs that are connected to a total of " 
     << nfeds << " FEDs" << std::endl
     << " " << detected_.size() 
     << " APV pairs have been detected, but are not connected" << std::endl
     << " " << undetected_.size()
     << " APV pairs are undetected (wrt DCU-DetId map)" << std::endl
     << " " << cntr
     << " FED channels out of a possible " << (96*nfeds)
     << " (" << nfeds << " FEDs) are unconnected (" 
     << percent << "%)" << std::endl
     << std::endl;
  
}

// -----------------------------------------------------------------------------
// 
void SiStripFedCabling::terse( std::stringstream& ss ) const {
  

  ss << "[SiStripFedCabling::" << __func__ << "]";
    
  if ( feds_.empty() ) {
    ss << " No FEDs found! Unable to print cabling map!";
    return;
  } 
  
  ss << " Printing cabling map for " << feds_.size()
     << " FEDs: " << std::endl << std::endl;
  
  std::vector<uint16_t>::const_iterator ifed = feds_.begin(); 
  std::vector<uint16_t>::const_iterator jfed = feds_.end(); 
  for ( ; ifed != jfed; ++ifed ) {
    
    uint16_t index = *ifed - FEDNumbering::MINSiStripFEDID;
    if ( index < registry_.size() ) { 
      ConnsRange conns = range( registry_[ index ] ); 
      
      ss << " Printing cabling information for FED id " << *ifed 
	 << " (found " << conns.size() 
	 << " FedChannelConnection objects...)"
	 << std::endl;
      
      uint16_t connected = 0;
      ConnsConstIter iconn = conns.begin();
      ConnsConstIter jconn = conns.end();
      for ( ; iconn != jconn; ++iconn ) { 
	if ( iconn->fedId() != sistrip::invalid_ ) { 
	  connected++; 
	  iconn->terse(ss); 
	  ss << std::endl;
	}
      } 
      
      ss << " Found " << connected 
	 << " connected channels for FED id " << *ifed << std::endl
	 << std::endl;
      
    }
    
  }
  
}

// -----------------------------------------------------------------------------
//
void SiStripFedCabling::printSummary( std::stringstream& ss, const TrackerTopology* /*trackerTopo*/ ) const {

  ss << "[SiStripFedCabling::" << __func__ << "]";
  
  if ( feds_.empty() ) {
    ss << " No FEDs found!";
    return;
  } 
  
  ss << " Found " << feds_.size() << " FEDs"
     << " with number of connected channels per front-end unit: " 
     << std::endl
     << " FedId FeUnit1 FeUnit2 FeUnit3 FeUnit4 FeUnit5 FeUnit6 FeUnit7 FeUnit8 Total" 
     << std::endl;
  
  uint16_t total = 0;
  uint16_t nfeds = 0;
  
  // iterate through fed ids
  std::vector<uint16_t>::const_iterator ii = feds_.begin(); 
  std::vector<uint16_t>::const_iterator jj = feds_.end(); 
  for ( ; ii != jj; ++ii ) { 

    // check number of connection objects
    uint16_t index = *ii - FEDNumbering::MINSiStripFEDID;
    if ( index < registry_.size() ) { 
      ConnsRange conns = range( registry_[ index ] ); 

      if ( conns.size() < 96 ) { 
	edm::LogError(mlCabling_) 
	  << "[SiStripFedCabling::" << __func__ << "]"
	  << " Unexpected size for FedChannelConnection vector! " 
	  << conns.size();
	return;
      }

      // count connected channels at level of fe unit
      std::vector<uint16_t> connected;
      connected.resize(8,0);
      for ( uint16_t ichan = 0; ichan < 96; ++ichan ) {
	ConnsConstIter iconn = conns.begin() + ichan;
	if ( iconn->fedId() < sistrip::valid_ ) { 
	  uint16_t unit = SiStripFedKey::feUnit(ichan);
	  if ( unit > 8 ) { continue; }
	  connected[unit-1]++; 
	} 
      }
      
      // increment counters
      uint16_t tot = 0 ;
      ss << " " << std::setw(5) << *ii;
      if ( !connected.empty() ) { nfeds++; }
      for ( uint16_t unit = 0; unit < 8; ++unit ) {
	ss << " " << std::setw(7) << connected[unit];
	if ( !connected.empty() ) { tot += connected[unit]; }
      }
      ss << " " << std::setw(5) << tot << std::endl;
      total += tot;
      
    } 
  
  }
  
  // print out
  float percent = (100.*total) / (96.*nfeds);
  percent = static_cast<uint16_t>( 10.*percent );
  percent /= 10.;
  ss << " Found: " << std::endl 
     << " " << nfeds << " out of " << feds_.size()
     << " FEDs with at least one connected channel " << std::endl 
     << " " << feds_.size() - nfeds << " out of " << feds_.size()
     << " FEDs with no connected channels." << std::endl 
     << " " << total << " connected channels in total" << std::endl
     << " " << detected_.size() 
     << " APV pairs have been detected, but are not connected" << std::endl
     << " " << undetected_.size() 
     << " APV pairs are undetected (wrt DCU-DetId map)" << std::endl
     << " " << percent
     << "% of FED channels are connected"  << std::endl;
  
}
