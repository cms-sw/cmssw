#ifndef CondFormats_SiStripObjects_FedChannelConnection_H
#define CondFormats_SiStripObjects_FedChannelConnection_H

#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include <boost/cstdint.hpp>
#include <ostream>
#include <sstream>

class FedChannelConnection;

/** Debug info for FedChannelConnection class. */
std::ostream& operator<< ( std::ostream&, const FedChannelConnection& );

/** 
    @class FedChannelConnection 
    @author R.Bainbridge

    @brief Class containning control, module, detector and connection
    information, at the level of a FED channel.
*/
class FedChannelConnection {
  
 public:

  // ---------- Constructor ----------

  /** Constructor requires at least information to uniquely identify a
   front-end module: ie, crate + FEC + ring + CCU + module. */
  FedChannelConnection( const uint16_t& fec_crate, 
			const uint16_t& fec_slot, 
			const uint16_t& fec_ring, 
			const uint16_t& ccu_addr, 
			const uint16_t& ccu_chan, 
			const uint16_t& apv0 = 0,
			const uint16_t& apv1 = 0,
			const uint32_t& dcu_id = 0,
			const uint32_t& det_id = 0,
			const uint16_t& pairs  = 0,
			const uint16_t& fed_id = 0,
			const uint16_t& fed_ch = 0,
			const uint16_t& length = 0,
			const bool& dcu = false,
			const bool& pll = false,
			const bool& mux = false,
			const bool& lld = false );
  
  /** Default constructor. */
  FedChannelConnection();
  
  /** Default destructor. */
  ~FedChannelConnection() {;}

  // ---------- Utility methods ----------
  
  /** Identifies a valid connection. */
  inline bool isConnected() const;
  
  /** Performs consistency check for this connection object. */
  void consistencyCheck() const {;} //@@ TO BE IMPLEMENTED...
  
  /** Prints all information for this connection object. */
  void print( std::stringstream& ) const;
  
  // -------------------- Module and detector information --------------------

  /** Returns DCUid for this module. */
  inline const uint32_t& dcuId() const;

  /** Returns DetId for this module. */
  inline const uint32_t& detId() const;

  /** Returns number of detector strips for this module. */
  inline uint16_t nDetStrips() const;

  /** Returns number of APV pairs for this module. */
  inline const uint16_t& nApvPairs() const;

  /** Returns number of APVs for this module. */
  inline uint16_t nApvs() const;
  
  // -------------------- FED connection information --------------------

  /** Returns APV pair number for this connection object. This can be
      either 0->1 or 0->2, depending on number of detector strips. */
  uint16_t apvPairNumber() const;

  /** Returns Laser Driver channel (1->3) for this channel. */
  uint16_t lldChannel() const;
  
  /** Returns FED id for this channel. */
  inline const uint16_t& fedId() const;

  /** Returns FED id for this channel. */
  inline const uint16_t& fedCh() const;

  /** Sets FED id for this channel. */
  inline void fedId( uint16_t& fed_id );

  /** Sets FED id for this channel. */
  inline void fedCh( uint16_t& fed_ch );

  // -------------------- Control structure information --------------------

  /** Returns FEC crate number. */ 
  inline const uint16_t& fecCrate() const;

  /** Returns slot number of FEC. */
  inline const uint16_t& fecSlot() const;

  /** Returns FEC ring number. */
  inline const uint16_t& fecRing() const;

  /** Returns CCU address. */
  inline const uint16_t& ccuAddr() const;

  /** Returns CCU channel. */
  inline const uint16_t& ccuChan() const;

  // -------------------- Front-end ASICs --------------------

  /** Indicates whether APV0 or APV1 of the pair has been found: a
      non-zero value indicates the I2C address; a null value signifies
      a problematic APV. */
  const uint16_t& i2cAddr( const uint16_t& apv0_or_1 ) const; 
  
  /** Indicates whether DCU ASIC is found. */ 
  inline const bool& dcu() const;

  /** Indicates whether APV-MUX ASIC is found. */ 
  inline const bool& mux() const;

  /** Indicates whether PLL ASIC is found. */ 
  inline const bool& pll() const;

  /** Indicates whether Linear Laser Driver ASIC is found. */ 
  inline const bool& lld() const;
  
 private:
  
  // ---------- Private member data ----------
  
  // Control
  uint16_t fecCrate_;
  uint16_t fecSlot_;
  uint16_t fecRing_;
  uint16_t ccuAddr_;
  uint16_t ccuChan_;

  // I2C addresses
  uint16_t apv0_; 
  uint16_t apv1_; 

  // Module / Detector
  uint32_t dcuId_;
  uint32_t detId_;
  uint16_t nApvPairs_;

  // FED
  uint16_t fedId_;
  uint16_t fedCh_;
  uint16_t length_;

  // Found devices
  bool dcu0x00_; 
  bool mux0x43_; 
  bool pll0x44_; 
  bool lld0x60_; 

};

// ---------- Inline methods ----------

bool FedChannelConnection::isConnected() const { 
  return ( fedId_ != sistrip::invalid_ && 
	   fedCh_ != sistrip::invalid_ );
}

const uint16_t& FedChannelConnection::fecCrate() const { return fecCrate_; } 
const uint16_t& FedChannelConnection::fecSlot() const { return fecSlot_; } 
const uint16_t& FedChannelConnection::fecRing() const { return fecRing_; }
const uint16_t& FedChannelConnection::ccuAddr() const { return ccuAddr_; }
const uint16_t& FedChannelConnection::ccuChan() const { return ccuChan_; }

const bool& FedChannelConnection::dcu() const { return dcu0x00_; }
const bool& FedChannelConnection::mux() const { return mux0x43_; }
const bool& FedChannelConnection::pll() const { return pll0x44_; }
const bool& FedChannelConnection::lld() const { return lld0x60_; }

const uint32_t& FedChannelConnection::dcuId() const { return dcuId_; }
const uint32_t& FedChannelConnection::detId() const { return detId_; }
uint16_t FedChannelConnection::nDetStrips() const { return 256*nApvPairs_; }
const uint16_t& FedChannelConnection::nApvPairs() const { return nApvPairs_; }
uint16_t FedChannelConnection::nApvs() const { return 2*nApvPairs(); }

const uint16_t& FedChannelConnection::fedId() const { return fedId_; }
const uint16_t& FedChannelConnection::fedCh() const { return fedCh_; }

void FedChannelConnection::fedId( uint16_t& fed_id ) { fedId_ = fed_id; }
void FedChannelConnection::fedCh( uint16_t& fed_ch ) { fedCh_ = fed_ch; }

#endif // CondFormats_SiStripObjects_FedChannelConnection_H

