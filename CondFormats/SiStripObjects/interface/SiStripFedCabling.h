#ifndef CondFormats_SiStripObjects_SiStripFedCabling_H
#define CondFormats_SiStripObjects_SiStripFedCabling_H

#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include <boost/cstdint.hpp>
#include <vector>

/** 
    \class SiStripFedCabling 
    \brief Contains cabling info at the device level, including
    DetId, APV pair numbers, hardware addresses, DCU id...
*/
class SiStripFedCabling {
  
 public:
  
  /** Constructor taking FED channel connection objects as input. */
  SiStripFedCabling( const std::vector<FedChannelConnection>& connections );
  /** Virtual destructor. */
  virtual ~SiStripFedCabling();

  /** Active FEDs. */
  const std::vector<uint16_t>& feds() const;
  /** Connection info for FE devices connected to a given FED id and channel. */
  const FedChannelConnection& connection( uint16_t fed_id,
					  uint16_t fed_chan ) const; 
  /** Connection info for FE devices connected to a given FED. */
  const std::vector<FedChannelConnection>& connections( uint16_t fed_id ) const; 
  
 private:
  
  /** Private default constructor. */
  SiStripFedCabling() {;}

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

#endif // CondFormats_SiStripObjects_SiStripFedCabling_H

