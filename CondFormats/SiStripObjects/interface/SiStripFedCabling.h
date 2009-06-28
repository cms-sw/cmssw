#ifndef CondFormats_SiStripObjects_SiStripFedCabling_H
#define CondFormats_SiStripObjects_SiStripFedCabling_H

#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include <boost/range/iterator_range.hpp>
#include <boost/cstdint.hpp>
#include <sstream>
#include <vector>
#include <string>

#define SISTRIPCABLING_USING_NEW_STRUCTURE
#define SISTRIPCABLING_USING_NEW_INTERFACE


// -------------------------------------------------------
#ifndef SISTRIPCABLING_USING_NEW_STRUCTURE // ------------
// -------------------------------------------------------


class SiStripFedCabling;

/** Debug info for SiStripFedCabling class. */
std::ostream& operator<< ( std::ostream&, const SiStripFedCabling& );

/** 
    \class SiStripFedCabling 
    \brief Contains cabling info at the device level, including
    DetId, APV pair numbers, hardware addresses, DCU id...
*/
class SiStripFedCabling {
  
 public:
  
  // -------------------- Constructors, destructors --------------------

  /** Constructor taking FED channel connection objects as input. */
  SiStripFedCabling( const std::vector<FedChannelConnection>& );

  /** Copy constructor. */
  SiStripFedCabling( const SiStripFedCabling& ); 

  /** Public default constructor. */
  SiStripFedCabling();

  /** Default destructor. */
  ~SiStripFedCabling();

  // -------------------- Utility methods --------------------
  
  /** Builds FED cabling from vector of FED connections. */
  void buildFedCabling( const std::vector<FedChannelConnection>& connections );
  
  /** Prints all connection information for this FED cabling object. */
  void print( std::stringstream& ) const;
  
  /** Prints terse information for this FED cabling object. */
  void terse( std::stringstream& ) const;
  
  /** Prints summary information for this FED cabling object. */
  void summary( std::stringstream& ) const;

  // -------------------- Methods to retrieve connections --------------------

  /** Retrieve vector of active FED ids. */
  const std::vector<uint16_t>& feds() const;
  
  /** Returns all connection info for a given FED id. */
  const std::vector<FedChannelConnection>& connections( uint16_t fed_id ) const; 
  
  /** Returns Connection info for a given FED id and channel. */
  const FedChannelConnection& connection( uint16_t fed_id,
					  uint16_t fed_chan ) const; 
  
  /** Returns information for "detected, but unconnected" devices. */
  inline const std::vector<FedChannelConnection>& detected() const; 
  
  /** Returns information for all "undetected" devices. */
  inline const std::vector<FedChannelConnection>& undetected() const; 
  
  // -------------------- Private member data --------------------

 private:
  
  /** "Active" FEDs that have connected FE devices. */
  std::vector<uint16_t> feds_;

  /** 
      Channel-level connection information for FE devices that: 
      - have been detected (ie, have non-zero FEC-related fields),
      - have been connected to a FED channel,
      - have a DcuId/DetId or NOT (=> cannot be used by recon sw).
      Info is arranged according to FED id and channel. 
      (1st index is FED id, 2nd index is FED channel.) 
  */
  std::vector< std::vector<FedChannelConnection> > connected_;

  /** 
      Channel-level connection information for FE devices that: 
      - have been detected (ie, have non-zero FEC-related fields),
      - have NOT been connected to a FED channel,
      - have OR do not have a DcuId/DetId.
  */
  std::vector<FedChannelConnection> detected_;

  /** 
      Channel-level connection information for FE devices that: 
      - have NOT been detected (ie, have zero FEC-related fields),
      - have NOT been connected to a FED channel,
      - do NOT have a DCU id.
      The DetId for these devices are inferred from the static LUT in
      the configuration database.
  */
  std::vector<FedChannelConnection> undetected_;

};

const std::vector<FedChannelConnection>& SiStripFedCabling::detected() const { return detected_; }
const std::vector<FedChannelConnection>& SiStripFedCabling::undetected() const{ return undetected_; }


// -------------------------------------------------------------
#else // SISTRIPCABLING_USING_NEW_STRUCTURE --------------------
#ifndef SISTRIPCABLING_USING_NEW_INTERFACE // ------------------
// -------------------------------------------------------------


class SiStripFedCabling;

/** Debug info for SiStripFedCabling class. */
std::ostream& operator<< ( std::ostream&, const SiStripFedCabling& );

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
  SiStripFedCabling( const std::vector<FedChannelConnection>& );

  /** Copy constructor. */
  SiStripFedCabling( const SiStripFedCabling& ); 

  /** Public default constructor. */
  SiStripFedCabling();

  /** Default destructor. */
  ~SiStripFedCabling();

  // -------------------- Methods to retrieve connections --------------------

  /** Retrieve vector of active FED ids. */
  const std::vector<uint16_t>& feds() const;
  
  /** Returns all connection info for a given FED id. */
  const std::vector<FedChannelConnection>& connections( uint16_t fed_id ) const; 
  
  /** Returns Connection info for a given FED id and channel. */
  const FedChannelConnection& connection( uint16_t fed_id,
					  uint16_t fed_chan ) const; 
  
  /** Returns information for "detected, but unconnected" devices. */
  inline const std::vector<FedChannelConnection>& detected() const; 
  
  /** Returns information for all "undetected" devices. */
  inline const std::vector<FedChannelConnection>& undetected() const; 

  // -------------------- Utility methods --------------------
  
  /** Builds FED cabling from vector of FED connections. */
  void buildFedCabling( const std::vector<FedChannelConnection>& connections );
  
  /** Prints all connection information for this FED cabling object. */
  void print( std::stringstream& ) const;
  
  /** Prints terse information for this FED cabling object. */
  void terse( std::stringstream& ) const;
  
  /** Prints summary information for this FED cabling object. */
  void summary( std::stringstream& ) const;

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

};

std::ostream& operator<<( std::ostream&, const SiStripFedCabling::ConnsRange& );

inline const std::vector<FedChannelConnection>& SiStripFedCabling::detected() const { 
  return detected_; 
}

inline const std::vector<FedChannelConnection>& SiStripFedCabling::undetected() const{ 
  return undetected_; 
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


// -------------------------------------------------------------
#else // SISTRIPCABLING_USING_NEW_INTERFACE --------------------
// -------------------------------------------------------------


class SiStripFedCabling;

/** Debug info for SiStripFedCabling class. */
std::ostream& operator<< ( std::ostream&, const SiStripFedCabling& );

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
  
  // -------------------- TO BE DEPRECATED! --------------------
  
  /** TO BE DEPRECATED! TO BE DEPRECATED! TO BE DEPRECATED! */
  SiStripFedCabling( const std::vector<FedChannelConnection>& );
  
  /** TO BE DEPRECATED! TO BE DEPRECATED! TO BE DEPRECATED! */
  void buildFedCabling( const std::vector<FedChannelConnection>& );
  
  /** TO BE DEPRECATED! TO BE DEPRECATED! TO BE DEPRECATED! */
  const std::vector<FedChannelConnection>& connections( uint16_t fed_id ) const; 
  
  /** TO BE DEPRECATED! TO BE DEPRECATED! TO BE DEPRECATED! */
  const FedChannelConnection& connection( uint16_t fed_id, uint16_t fed_ch ) const; 
  
  /** TO BE DEPRECATED! TO BE DEPRECATED! TO BE DEPRECATED! */
  const std::vector<uint16_t>& feds() const;
  
  /** TO BE DEPRECATED! TO BE DEPRECATED! TO BE DEPRECATED! */
  const std::vector<FedChannelConnection>& detected() const; 
  
  /** TO BE DEPRECATED! TO BE DEPRECATED! TO BE DEPRECATED! */
  const std::vector<FedChannelConnection>& undetected() const; 
  
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
  void printDebug( std::stringstream& ) const;

  /// LEFT FOR COMPATIBILITY. SHOULD BE REPLACED BY PRINTDEBUG
  void print( std::stringstream& ss ) const {
    printDebug(ss);
  }
  
  /** Prints terse information for this FED cabling object. */
  void terse( std::stringstream& ) const;
  
  /** Prints summary information for this FED cabling object. */
  void printSummary( std::stringstream& ) const;
  /// LEFT FOR COMPATIBILITY. SHOULD BE REPLACED BY PRINTSUMMARY
  void summary( std::stringstream& ss ) const {
    printSummary(ss);
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


// -------------------------------------------------------------
#endif // SISTRIPCABLING_USING_NEW_INTERFACE -------------------
#endif // SISTRIPCABLING_USING_NEW_STRUCTURE -------------------
// -------------------------------------------------------------

#endif // CondFormats_SiStripObjects_SiStripFedCabling_H

