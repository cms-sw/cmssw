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
    dcuId_(0), detId_(0), nPairs_(0),
    cabling_()  {;}
  
  ~SiStripModule() {;}
  
  inline const uint16_t& ccuChan() const { return ccuChan_; }
  void addDevices( const FedChannelConnection& conn );
  
  // getters
  inline const uint32_t& dcuId() const { return dcuId_; } 
  inline const uint32_t& detId() const { return detId_; } 
  inline const uint16_t& nPairs() const { return nPairs_; }
  inline const uint16_t& dcu() const { return dcu0x00_; } 
  inline const uint16_t& mux() const { return mux0x43_; } 
  inline const uint16_t& pll() const { return pll0x44_; } 
  inline const uint16_t& lld() const { return lld0x60_; } 
  vector<uint16_t> apvs();
  uint16_t apv( uint16_t apv_id )  const;
/*   pair<uint16_t,uint16_t> pair( uint16_t apv_pair ) const; */
/*   pair<uint16_t,uint16_t> fedCh( uint16_t apv_pair ) const; */
  
  // setters
  inline void dcuId( const uint32_t& dcu_id ) { if ( !dcuId_ && dcu_id ) { dcuId_ = dcu_id; dcu0x00_ = true; } }
  inline void detId( const uint32_t& det_id ) { if ( !detId_ && det_id ) { detId_ = det_id; } } 
  inline void nPairs( const uint16_t& npairs ) { if ( !nPairs_ && npairs ) { nPairs_ = npairs; } } 
  
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
  uint16_t nPairs_;
  
  // FED cabling: KEY = APV I2C address, DATA = <FedId,FedCh>
  map< uint16_t, pair<uint16_t,uint16_t> > cabling_;
  
};

// -----------------------------------------------------------------------------
class SiStripCcu {
  
 public: 
  
  SiStripCcu( const FedChannelConnection& conn ) : ccuAddr_( conn.ccuAddr() ), modules_() {;}
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

  SiStripRing( const FedChannelConnection& conn ) : fecRing_( conn.fecRing() ), ccus_() {;}
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

  SiStripFec( const FedChannelConnection& conn ) : fecSlot_( conn.fecSlot() ), rings_() {;}
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
  inline void fedCh( const FedChannelConnection& conn );

  // misc
  void countDevices( uint32_t& fecs,
		     uint32_t& rings,
		     uint32_t& ccus,
		     uint32_t& modules,
		     uint32_t& apvs,
		     uint32_t& dcuids,
		     uint32_t& detids,
		     uint32_t& npairs,
		     uint32_t& fedchans,
		     uint32_t& dcus,
		     uint32_t& muxes,
		     uint32_t& plls,
		     uint32_t& llds );
  
 private:

  SiStripModule& module( const FedChannelConnection& conn );
  vector<SiStripFec> fecs_;
  
};

void SiStripFecCabling::dcuId( const FedChannelConnection& conn ) { 
  module(conn).dcuId(conn.dcuId()); 
}

void SiStripFecCabling::detId( const FedChannelConnection& conn ) { 
  module(conn).detId(conn.dcuId()); 
}

void SiStripFecCabling::fedCh( const FedChannelConnection& conn ) { 
  module(conn).detId(conn.fedId()); 
  module(conn).detId(conn.fedCh()); 
}


#endif // CalibTracker_SiStripObjects_SiStripFecCabling_H


