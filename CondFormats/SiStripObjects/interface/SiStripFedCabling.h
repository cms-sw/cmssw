#ifndef CondFormats_SiStripObjects_SiStripFedCabling_H
#define CondFormats_SiStripObjects_SiStripFedCabling_H

#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include <boost/cstdint.hpp>
#include <sstream>
#include <vector>
#include <string>

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

// -------------------- Inline methods --------------------

const std::vector<FedChannelConnection>& SiStripFedCabling::detected() const { return detected_; }
const std::vector<FedChannelConnection>& SiStripFedCabling::undetected() const{ return undetected_; }

#endif // CondFormats_SiStripObjects_SiStripFedCabling_H

