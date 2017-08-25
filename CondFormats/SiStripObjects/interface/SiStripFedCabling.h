#ifndef CondFormats_SiStripObjects_SiStripFedCabling_H
#define CondFormats_SiStripObjects_SiStripFedCabling_H

#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include <boost/range/iterator_range.hpp>
#include <boost/cstdint.hpp>
#include <sstream>
#include <vector>
#include <string>

#define SISTRIPCABLING_USING_NEW_STRUCTURE
#define SISTRIPCABLING_USING_NEW_INTERFACE

class TrackerTopology;

class SiStripFedCabling;

/** 
    \class SiStripFedCabling 
    \brief Contains cabling info at the device level, including
    DetId, APV pair numbers, hardware addresses, DCU id...
*/
class SiStripFedCabling {
  
 public:
  
  // -------------------- Typedefs and structs --------------------

  typedef std::vector<uint16_t> Feds;

  typedef Feds::iterator FedsIter;

  typedef Feds::const_iterator FedsConstIter;

  typedef boost::iterator_range<FedsIter> FedsIterRange;

  typedef boost::iterator_range<FedsConstIter> FedsConstIterRange;

  typedef std::vector<FedChannelConnection> Conns;

  typedef Conns::iterator ConnsIter;

  typedef Conns::const_iterator ConnsConstIter;
  
  typedef boost::iterator_range<ConnsIter> ConnsIterRange;

  typedef boost::iterator_range<ConnsConstIter> ConnsConstIterRange;

  typedef std::pair<uint32_t,uint32_t> ConnsPair;
  
  typedef std::vector<ConnsPair> Registry;
  
  // -------------------- Constructors, destructors --------------------
  
  /** Constructor taking FED channel connection objects as input. */
  SiStripFedCabling( ConnsConstIterRange );
  
  /** Copy constructor. */
  SiStripFedCabling( const SiStripFedCabling& ); 
  
  /** Public default constructor. */
  SiStripFedCabling();

  /** Default destructor. */
  ~SiStripFedCabling();

  // -------------------- Methods to retrieve connections --------------------

  /** Retrieve vector of active FED ids. */
  FedsConstIterRange fedIds() const;
  
  /** Returns all connection objects for a given FED id. */
  ConnsConstIterRange fedConnections( uint16_t fed_id ) const; 
  
  /** Returns connection object for a given FED id and channel. */
  FedChannelConnection fedConnection( uint16_t fed_id, uint16_t fed_ch ) const; 
  
  /** Returns information for "detected, but unconnected" devices. */
  ConnsConstIterRange detectedDevices() const; 
  
  /** Returns information for all "undetected" devices. */
  ConnsConstIterRange undetectedDevices() const; 

  // -------------------- Utility methods --------------------
  
  /** Builds FED cabling from vector of FED connections. */
  void buildFedCabling( ConnsConstIterRange connections );
  
  /** Prints all connection information for this FED cabling object. */
  void printDebug( std::stringstream&, const TrackerTopology* trackerTopo ) const;

  /// LEFT FOR COMPATIBILITY. SHOULD BE REPLACED BY PRINTDEBUG
  void print( std::stringstream& ss, const TrackerTopology* trackerTopo ) const {
    printDebug(ss, trackerTopo);
  }
  
  /** Prints terse information for this FED cabling object. */
  void terse( std::stringstream& ) const;
  
  /** Prints summary information for this FED cabling object. */
  void printSummary( std::stringstream&, const TrackerTopology* trackerTopo ) const;
  /// LEFT FOR COMPATIBILITY. SHOULD BE REPLACED BY PRINTSUMMARY
  void summary( std::stringstream& ss, const TrackerTopology* trackerTopo ) const {
    printSummary(ss, trackerTopo);
  }

  /// Builds range of iterators from pair of offsets
  class ConnsRange {

  public:
    
    ConnsRange( const Conns&, ConnsPair );
    ConnsRange( const Conns& );
    ConnsRange() {;}
    
    ConnsConstIter begin() const;
    ConnsConstIter end() const;
    ConnsConstIterRange range() const;
    ConnsConstIterRange invalid() const;
    
    bool empty() const;
    uint32_t size() const;

    ConnsPair connsPair() const;
    static ConnsPair emptyPair();

    void print( std::stringstream& ) const;
    
  private:
    
    ConnsConstIterRange vector_;
    ConnsConstIterRange range_;

  };

  /// Builds range of iterators from pair of offsets
  ConnsRange range( ConnsPair ) const;
  
  // -------------------- Private member data --------------------

 private:

  /// "Active" FEDs that have connected FE devices
  Feds feds_;
  
  /// Container of "ranges" indexed by FED id 
  Registry registry_;
  
  /// Container of connection objects 
  Conns connections_;

  /// Connections to FE devices that are not detected
  Conns detected_;
  
  /// FE devices that are detected
  Conns undetected_;


 COND_SERIALIZABLE;
};

std::ostream& operator<<( std::ostream&, const SiStripFedCabling::ConnsRange& );

inline SiStripFedCabling::FedsConstIterRange SiStripFedCabling::fedIds() const {
  return FedsConstIterRange( feds_.begin(), feds_.end() );
}

inline SiStripFedCabling::ConnsConstIterRange SiStripFedCabling::detectedDevices() const { 
  return ConnsConstIterRange( detected_.begin(), detected_.end() ); 
}

inline SiStripFedCabling::ConnsConstIterRange SiStripFedCabling::undetectedDevices() const{ 
  return ConnsConstIterRange( undetected_.begin(), undetected_.end() ); 
}

inline SiStripFedCabling::ConnsConstIter SiStripFedCabling::ConnsRange::begin() const { 
  return range_.begin(); 
}

inline SiStripFedCabling::ConnsConstIter SiStripFedCabling::ConnsRange::end() const { 
  return range_.end(); 
}

inline SiStripFedCabling::ConnsConstIterRange SiStripFedCabling::ConnsRange::range() const { 
  return range_;
}

inline SiStripFedCabling::ConnsConstIterRange SiStripFedCabling::ConnsRange::invalid() const { 
  return ConnsConstIterRange( vector_.end(), vector_.end() );
}

inline bool SiStripFedCabling::ConnsRange::empty() const { 
  return ( range_.begin() == range_.end() );
}

inline uint32_t SiStripFedCabling::ConnsRange::size() const { 
  return std::distance( range_.begin(), range_.end() );
}

inline SiStripFedCabling::ConnsPair SiStripFedCabling::ConnsRange::connsPair() const { 
  return ( ( range_.begin() == vector_.end() && 
	     range_.end()   == vector_.end() ) ?
	   ConnsPair( sistrip::invalid32_, sistrip::invalid32_ ) :
	   ConnsPair( std::distance( vector_.begin(), range_.begin() ), 
		      std::distance( vector_.begin(), range_.end() ) ) );
}

inline SiStripFedCabling::ConnsPair SiStripFedCabling::ConnsRange::emptyPair() { 
  return ConnsPair( sistrip::invalid32_, sistrip::invalid32_ );
}

inline SiStripFedCabling::ConnsRange SiStripFedCabling::range( SiStripFedCabling::ConnsPair p ) const {
  return ConnsRange( connections_, p );
}

#endif // CondFormats_SiStripObjects_SiStripFedCabling_H

