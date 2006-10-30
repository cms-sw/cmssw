// Last commit: $Id: SiStripModule.h,v 1.4 2006/10/10 13:06:48 bainbrid Exp $
// Latest tag:  $Name:  $
// Location:    $Source: /cvs_server/repositories/CMSSW/CMSSW/CalibFormats/SiStripObjects/interface/SiStripModule.h,v $

#ifndef CalibFormats_SiStripObjects_SiStripModule_H
#define CalibFormats_SiStripObjects_SiStripModule_H

#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include <boost/cstdint.hpp>
#include <sstream>
#include <vector>
#include <map>

class SiStripModule;

/** Debug info for SiStripModule class. */
std::ostream& operator<< ( std::ostream&, const SiStripModule& );

/** 
    @class SiStripModule
    @author R.Bainbridge
    @brief Device and connection information at the level of a
    front-end module.
*/
class SiStripModule {
  
 public: 

  // ---------- Constructors and adding devices ----------
  
  /** */
  SiStripModule( const FedChannelConnection& conn ) 
    : path_( conn.fecCrate(), 
	     conn.fecSlot(), 
	     conn.fecRing(), 
	     conn.ccuAddr(), 
	     conn.ccuChan() ),
    apv0x32_(0), apv0x33_(0), apv0x34_(0), apv0x35_(0), apv0x36_(0), apv0x37_(0), 
    dcu0x00_(0), mux0x43_(0), pll0x44_(0), lld0x60_(0),
    dcuId_(0), detId_(0), nApvPairs_(0),
    cabling_(), length_(0) { addDevices( conn ); }

  /** Default constructor. */
  ~SiStripModule() {;}

  /** Sets device info (addresses, DetID, etc) for this module. */
  void addDevices( const FedChannelConnection& conn );

  // ---------- Typedefs and enums ----------

  /** Pair containing FED id/channel. */
  typedef std::pair<uint16_t,uint16_t> PairOfU16;
  
  /** Pair containing FED id/channel. */
  typedef PairOfU16 FedChannel;

  /** Map between LLD channel and FED channel */
  typedef std::map< uint16_t, FedChannel > FedCabling;

  // ---------- Control structure ----------
  
  inline const uint16_t& fecCrate() const;
  inline const uint16_t& fecSlot() const;
  inline const uint16_t& fecRing() const;
  inline const uint16_t& ccuAddr() const;
  inline const uint16_t& ccuChan() const;
  
  /** Returns control "path" for this module, containing address
      information on FEC crate, slot, ring, CCU, and module. */
  inline const SiStripFecKey::Path& path() const;
  
  // ---------- APV devices ----------

  /** Returns I2C addresses of active ("found") APVs. */
  std::vector<uint16_t> activeApvs() const;
  
  /** Identifies whether APV of a given I2C address (32->37) or
      footprint position on the hybrid (0->5) is active or
      not. Returns device I2C address or zero if not active. */
  const uint16_t& activeApv( const uint16_t& apv_address ) const;
  
  /** Identifies APV pairs that are active, for a given LLD channel
      (0->2). Returns device I2C address or zero if not active. */
  PairOfU16 activeApvPair( const uint16_t& lld_channel ) const;

  /** Add APV to module using I2C address (32->37). */
  void addApv( const uint16_t& apv_address );

  // ---------- Other hybrid devices ----------
  
  /** Identifies whether the DCU device is active ("found") or not. */
  inline const uint16_t& dcu() const;

  /** Identifies whether the MUX device is active ("found") or not. */
  inline const uint16_t& mux() const;

  /** Identifies whether the PLL device is active ("found") or not. */
  inline const uint16_t& pll() const;

  /** Identifies whether the LLD device is active ("found") or not. */
  inline const uint16_t& lld() const;

  // ---------- Module information ----------
  
  /** Returns DCU id for this module. */
  inline const uint32_t& dcuId() const;

  /** Returns LLD channel (0->2) for a given APV pair number (0->1 or
      0->2, depending on number of APV pairs). */
  uint16_t lldChannel( const uint16_t& apv_pair_num ) const;

  /** Set DCU id for this module. */
  inline void dcuId( const uint32_t& dcu_id );

  // ---------- Detector information ----------
  
  /** Returns unique (geometry-based) identifier for this module. */
  inline const uint32_t& detId() const;
  
  /** Returns APV pair number (0->1 or 0->2, depending on number of
      APV pairs) for a given LLD channel (0->2). */
  uint16_t apvPairNumber( const uint16_t& lld_channel ) const;

  /** Returns number of APV pairs for this module. */
  inline const uint16_t& nApvPairs() const;
  
  /** Returns number of detector strips for this module. */
  inline uint16_t nDetStrips() const;
  
  /** Set DetId for this module. */
  inline void detId( const uint32_t& det_id );

  /** Set number of detector strips for this module. */
  void nApvPairs( const uint16_t& npairs );
  
  // ---------- FED connection information ----------
  
  /** Returns map of apvPairNumber and FedChannel. */
  inline const FedCabling& fedChannels() const;
  
  /** Returns FedChannel for a given apvPairNumber. */
  const FedChannel& fedCh( const uint16_t& apv_pair_num ) const;
  
  /** Sets FedChannel for given APV address (32->37). Returns true
      if connection made, false otherwise. */
  bool fedCh( const uint16_t& apv_address, const FedChannel& fed_ch );
  
  // ---------- Miscellaneous ----------

  /** Prints some debug information for this module. */
  void print( std::stringstream& ) const; 

  /** Returns cable length. */
  inline const uint16_t& length() const;
  
  /** Sets cable length. */
  inline void length( const uint16_t& length );

 private: 
  
  /** Control "path" for this module. */
  SiStripFecKey::Path path_;
  
  // APVs found (with hex addr)  
  uint16_t apv0x32_;
  uint16_t apv0x33_;
  uint16_t apv0x34_;
  uint16_t apv0x35_;
  uint16_t apv0x36_;
  uint16_t apv0x37_;
  
  // Devices found (with hex addr)  
  uint16_t dcu0x00_;
  uint16_t mux0x43_;
  uint16_t pll0x44_;
  uint16_t lld0x60_;
  
  // Detector
  uint32_t dcuId_;
  uint32_t detId_;
  uint16_t nApvPairs_;
  
  // FED cabling: KEY = APV pair footprint position, DATA = FedId + FedCh
  FedCabling cabling_;
  uint16_t length_;
  
};

// --------------- inline methods ---------------

const uint16_t& SiStripModule::fecCrate() const { return path_.fecCrate_; } 
const uint16_t& SiStripModule::fecSlot() const { return path_.fecSlot_; } 
const uint16_t& SiStripModule::fecRing() const { return path_.fecRing_; }
const uint16_t& SiStripModule::ccuAddr() const { return path_.ccuAddr_; }
const uint16_t& SiStripModule::ccuChan() const { return path_.ccuChan_; }

const SiStripFecKey::Path& SiStripModule::path() const { return path_; }

const uint32_t& SiStripModule::dcuId() const { return dcuId_; } 
const uint32_t& SiStripModule::detId() const { return detId_; } 
const uint16_t& SiStripModule::nApvPairs() const { return nApvPairs_; }
uint16_t SiStripModule::nDetStrips() const { return 256*nApvPairs_; }

void SiStripModule::dcuId( const uint32_t& dcu_id ) { if ( dcu_id ) { dcuId_ = dcu_id; dcu0x00_ = true; } }
void SiStripModule::detId( const uint32_t& det_id ) { if ( det_id ) { detId_ = det_id; } } 
const SiStripModule::FedCabling& SiStripModule::fedChannels() const { return cabling_; } 

const uint16_t& SiStripModule::length() const { return length_; } 
void SiStripModule::length( const uint16_t& length ) { length_ = length; } 

const uint16_t& SiStripModule::dcu() const { return dcu0x00_; } 
const uint16_t& SiStripModule::mux() const { return mux0x43_; } 
const uint16_t& SiStripModule::pll() const { return pll0x44_; } 
const uint16_t& SiStripModule::lld() const { return lld0x60_; } 

#endif // CalibTracker_SiStripObjects_SiStripModule_H


