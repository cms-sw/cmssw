#ifndef CalibFormats_SiStripObjects_SiStripFecCabling_H
#define CalibFormats_SiStripObjects_SiStripFecCabling_H

#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include <boost/cstdint.hpp>
#include <vector>
#include <map>

/** 
    This file contains following classes:
    class SiStripModule;
    class SiStripCcu;
    class SiStripRing;
    class SiStripFec;
    class SiStripFecCabling;
*/

using namespace std;

// -----------------------------------------------------------------------------
class SiStripModule {
  
 public: 
  
  SiStripModule( const FedChannelConnection& conn ) 
    : ccuChan_( conn.ccuChan() ), 
    apv0x32_(0), apv0x33_(0), apv0x34_(0), apv0x35_(0), apv0x36_(0), apv0x37_(0), 
    dcu0x00_(0), mux0x43_(0), pll0x44_(0), lld0x60_(0),
    dcuId_(0), detId_(0), nApvPairs_(0),
    cabling_(), length_(0) { addDevices( conn ); }
  
  ~SiStripModule() {;}

  // ----- Misc -----

  /** Returns CCU channel for this module. */
  inline const uint16_t& ccuChan() const { return ccuChan_; }
  /** Sets device info (addresses, DetID, etc) for this module. */
  void addDevices( const FedChannelConnection& conn );
  /** Prints some debug information for this module. */
  void print() const; 
  
  // ----- APV addresses  -----

  /** Returns I2C addresses of active ("found") APVs. */
  vector<uint16_t> activeApvs() const;
  /** Identifies whether APV of a given I2C address (32->37) or
      footprint position on the hybrid (0->5) is active or
      not. Returns actual address instead of boolean. */
  const uint16_t& activeApv( const uint16_t& apv_address ) const;
  /** Identifies APV pairs that are active, for a given LLD channel
      (0->2). Returns actual address instead of boolean. */
  pair<uint16_t,uint16_t> activeApvPair( const uint16_t& lld_channel ) const;

  /** Add APV to module using I2C address (32->37). */
  void addApv( const uint16_t& apv_address );


  // ----- Detector/module information -----
  
  /** Returns DCU id for this module. */
  inline const uint32_t& dcuId() const { return dcuId_; } 
  /** Returns unique (geometry-based) identifier for this module. */
  inline const uint32_t& detId() const { return detId_; } 
  /** Returns number of detector strips for this module (and so allows
      to infer the number of APVs or APV pairs). */
  inline const uint16_t& nApvPairs() const { return nApvPairs_; }

  /** Returns LLD channel (0->2) for a given APV pair number (0->1 or
      0->2, depending on number of APV pairs). */
  uint16_t lldChannel( const uint16_t& apv_pair_num ) const;
  /** Returns APV pair number (0->1 or 0->2, depending on number of
      APV pairs) for a given LLD channel (0->2). */
  uint16_t apvPairNum( const uint16_t& lld_channel ) const;

  /** Set DCU id for this module. */
  inline void dcuId( const uint32_t& dcu_id ) { if ( dcu_id ) { dcuId_ = dcu_id; dcu0x00_ = true; } }
  /** Set DetId for this module. */
  inline void detId( const uint32_t& det_id ) { if ( det_id ) { detId_ = det_id; } } 
  /** Set number of detector strips for this module. */
  void nApvPairs( const uint16_t& npairs );
  
  // ----- FED information -----

  /** Returns map of APV pair (0->1 or 0->2) and FED id/channel. */
  inline const map< uint16_t, pair<uint16_t,uint16_t> >& fedChannels() const { return cabling_; } 
  /** Returns FED id/channel of a given APV pair (0->1 or 0->2). */
  const pair<uint16_t,uint16_t>& fedCh( const uint16_t& apv_pair_num ) const;

  /** Sets FED id/channel for given APV address (32->37). Returns true
      if connection made, false otherwise. */
  bool fedCh( const uint16_t& apv_address, const pair<uint16_t,uint16_t>& fed_ch );

  /** Returns cable length. */
  inline const uint16_t& length() const { return length_; } 
  /** Sets cable length. */
  inline void length( const uint16_t& length ) { length_ = length; } 

  // ----- Other hybrid devices -----
  
  /** Identifies whether the DCU device is active ("found") or not. */
  inline const uint16_t& dcu() const { return dcu0x00_; } 
  /** Identifies whether the MUX device is active ("found") or not. */
  inline const uint16_t& mux() const { return mux0x43_; } 
  /** Identifies whether the PLL device is active ("found") or not. */
  inline const uint16_t& pll() const { return pll0x44_; } 
  /** Identifies whether the LLD device is active ("found") or not. */
  inline const uint16_t& lld() const { return lld0x60_; } 

 private: 
  
  uint16_t ccuChan_;
  
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
  map< uint16_t, pair<uint16_t,uint16_t> > cabling_;
  uint16_t length_;
  
};

// -----------------------------------------------------------------------------
class SiStripCcu {
  
 public: 
  
  SiStripCcu( const FedChannelConnection& conn ) : ccuAddr_( conn.ccuAddr() ), modules_() { addDevices( conn ); }
  ~SiStripCcu() {;}
  
  inline const vector<SiStripModule>& modules() const { return modules_; }
  inline const uint16_t& ccuAddr() const { return ccuAddr_; }
  void addDevices( const FedChannelConnection& conn );
  
 private:

  SiStripCcu() {;}
  uint16_t ccuAddr_;
  vector<SiStripModule> modules_;

};

// -----------------------------------------------------------------------------
class SiStripRing {

 public: 

  SiStripRing( const FedChannelConnection& conn ) : fecRing_( conn.fecRing() ), ccus_() { addDevices( conn ); }
  ~SiStripRing() {;}

  inline const vector<SiStripCcu>& ccus() const { return ccus_; }
  inline const uint16_t& fecRing() const { return fecRing_; }
  void addDevices( const FedChannelConnection& conn );
  
 private:

  SiStripRing() {;}
  uint16_t fecRing_;
  vector<SiStripCcu> ccus_;

};

// -----------------------------------------------------------------------------
class SiStripFec {

 public: 

  SiStripFec( const FedChannelConnection& conn ) : fecSlot_( conn.fecSlot() ), rings_() { addDevices( conn ); }
  ~SiStripFec() {;}
  
  inline const vector<SiStripRing>& rings() const { return rings_; }
  inline const uint16_t& fecSlot() const { return fecSlot_; }
  void addDevices( const FedChannelConnection& conn );
  
 private:

  SiStripFec() {;}
  uint16_t fecSlot_;
  vector<SiStripRing> rings_;

};

// -----------------------------------------------------------------------------
/* 
   @class SiStripFecCabling
   @brief Brief description here!
*/
class SiStripFecCabling {
  
 public:

  SiStripFecCabling() {;}
  SiStripFecCabling( const SiStripFedCabling& );
  ~SiStripFecCabling() {;} //@@ needs implementation!!

  // getters
  inline const vector<SiStripFec>& fecs() const { return fecs_; }
  void connections( vector<FedChannelConnection>& ); //@@ gets all of them!
  
  // setters
  void addDevices( const FedChannelConnection& conn );
  inline void dcuId( const FedChannelConnection& conn );
  inline void detId( const FedChannelConnection& conn );
  //inline void fedCh( const FedChannelConnection& conn ); //@@ needs to be implemented
  inline void nApvPairs( const FedChannelConnection& conn );

  // misc
  void countDevices() const;

  const SiStripModule& module( const FedChannelConnection& conn ) const;
  
 private:

  vector<SiStripFec> fecs_;
  
};

 

void SiStripFecCabling::dcuId( const FedChannelConnection& conn ) { 
  const_cast<SiStripModule&>(module(conn)).dcuId(conn.dcuId()); 
}

void SiStripFecCabling::detId( const FedChannelConnection& conn ) { 
  const_cast<SiStripModule&>(module(conn)).detId(conn.detId()); 
}

void SiStripFecCabling::nApvPairs( const FedChannelConnection& conn ) { 
  const_cast<SiStripModule&>(module(conn)).nApvPairs(conn.nApvPairs()); 
}

/* void SiStripFecCabling::fedCh( const FedChannelConnection& conn ) {  */
/*   module(conn).detId(conn.fedId());  */
/*   module(conn).detId(conn.fedCh());  */
/* } */


#endif // CalibTracker_SiStripObjects_SiStripFecCabling_H


