#ifndef CalibFormats_SiStripObjects_SiStripFecCabling_H
#define CalibFormats_SiStripObjects_SiStripFecCabling_H

#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include <vector>
#include <map>

class SiStripFec;
class SiStripRing;
class SiStripCcu;
class SiStripModule;

using namespace std;

// -----------------------------------------------------------------------------
class SiStripFec {

 public: 

  SiStripFec( const FedChannelConnection& conn ) : fecSlot_( conn.fecSlot() ), rings_() {;}
  ~SiStripFec() {;}
  
  inline const vector<SiStripRing>& rings() const { return rings_; }
  inline const unsigned short& fecSlot() const { return fecSlot_; }
  void addDevices( const FedChannelConnection& conn );
  
 private:

  SiStripFec() {;}
  unsigned short fecSlot_;
  vector<SiStripRing> rings_;

};

// -----------------------------------------------------------------------------
class SiStripRing {

 public: 

  SiStripRing( const FedChannelConnection& conn ) : fecRing_( conn.fecRing() ), ccus_() {;}
  ~SiStripRing() {;}

  inline const vector<SiStripCcu>& ccus() const { return ccus_; }
  inline const unsigned short& fecRing() const { return fecRing_; }
  void addDevices( const FedChannelConnection& conn );
  
 private:

  SiStripRing() {;}
  unsigned short fecRing_;
  vector<SiStripCcu> ccus_;

};

// -----------------------------------------------------------------------------
class SiStripCcu {

 public: 
  
  SiStripCcu( const FedChannelConnection& conn ) : ccuAddr_( conn.ccuAddr() ), modules_() {;}
  ~SiStripCcu() {;}
  
  inline const vector<SiStripModule>& modules() const { return modules_; }
  inline const unsigned short& ccuAddr() const { return ccuAddr_; }
  void addDevices( const FedChannelConnection& conn );
  
 private:

  SiStripCcu() {;}
  unsigned short ccuAddr_;
  vector<SiStripModule> modules_;

};

// -----------------------------------------------------------------------------
class SiStripModule {

 public: 

  typedef pair<unsigned short, unsigned short> FedChannel;
  
  SiStripModule( const FedChannelConnection& conn ) 
    : ccuChan_( conn.ccuChan() ), 
    apvs_(), 
    dcu0x00_(false), mux0x43_(false), pll0x44_(false), lld0x60_(false),
    dcuId_(0), detId_(0), nPairs_(0),
    cabling_()  {;}

  ~SiStripModule() {;}
  
  inline const unsigned short& ccuChan() const { return ccuChan_; }
  void addDevices( const FedChannelConnection& conn );
  
  // getters
  inline const vector<unsigned short>& apvs() const { return apvs_; }
  inline const unsigned int& dcuId()          const { return dcuId_; } 
  inline const unsigned int& detId()          const { return detId_; } 
  inline const unsigned short& nPairs()       const { return nPairs_; }
  inline const bool& dcu() const { return dcu0x00_; } 
  inline const bool& mux() const { return mux0x43_; } 
  inline const bool& pll() const { return pll0x44_; } 
  inline const bool& lld() const { return lld0x60_; } 

  // setters
  inline void dcuId( const unsigned int& dcu_id )    { if ( !dcuId_ && dcu_id ) { dcuId_ = dcu_id; dcu0x00_ = true; } }
  inline void detId( const uint32_t& det_id )        { if ( !detId_ && det_id ) detId_ = det_id; } 
  inline void nPairs( const unsigned short& npairs ) { if ( !nPairs_ && npairs ) nPairs_ = npairs; } 
  
  //const SiStripApvPair& apvPair( unsigned short apv_pair ) const {;} //@@ needs implementation
  //void fedCh( const SiStripDevice& addr, unsigned short fed_id, unsigned short fed_chan );
  
 private: 
  
  SiStripModule() {;}
  unsigned short ccuChan_;

  // APVs found
  vector<unsigned short> apvs_;
  
  // Devices found (with hex addr)  
  bool dcu0x00_;
  bool mux0x43_;
  bool pll0x44_;
  bool lld0x60_;
    
  // Detector
  unsigned int   dcuId_;
  uint32_t       detId_;
  unsigned short nPairs_;

  // FED cabling: KEY = APV I2C address, DATA = <FedId,FedCh>
  map< unsigned short, FedChannel> cabling_;
  
};

/* 
   @class SiStripFecCabling
   @brief Brief description here!
*/
class SiStripFecCabling {
  
 public:

  SiStripFecCabling( const SiStripFedCabling& );
  ~SiStripFecCabling();

  inline const vector<SiStripFec>& fecs() const { return fecs_; }
  void addDevices( const FedChannelConnection& conn );
  inline void dcuId( const FedChannelConnection& conn );
  inline void detId( const FedChannelConnection& conn );
  inline void fedCh( const FedChannelConnection& conn );

  void countDevices( unsigned int& fecs,
		     unsigned int& rings,
		     unsigned int& ccus,
		     unsigned int& modules,
		     unsigned int& apvs,
		     unsigned int& dcuids,
		     unsigned int& detids,
		     unsigned int& npairs,
		     unsigned int& fedchans,
		     unsigned int& dcus,
		     unsigned int& muxes,
		     unsigned int& plls,
		     unsigned int& llds );
  
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


