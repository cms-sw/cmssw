#ifndef CondFormats_SiStripObjects_FedChannelConnection_H
#define CondFormats_SiStripObjects_FedChannelConnection_H

#include <boost/cstdint.hpp>
#include <vector>
#include <ostream>
#include <sstream>
#include <string>

class FedChannelConnection;

/** Debug info for FedChannelConnection class. */
std::ostream& operator<< ( std::ostream&, const FedChannelConnection& );

/** 
    @class FedChannelConnection 
    @brief FED channel-level device and connection information.
*/
class FedChannelConnection {
  
 public:
  
  FedChannelConnection()
    : fecCrate_(0), fecSlot_(0), fecRing_(0), ccuAddr_(0), ccuChan_(0),
    apv0_(0), apv1_(0), 
    dcuId_(0), detId_(0), nApvPairs_(0),
    fedId_(0), fedCh_(0), length_(0),
    dcu0x00_(false), mux0x43_(false), pll0x44_(false), lld0x60_(false) {;}
  
  /** Minimum information allowed is that required to uniquely
      identify a module (ie, crate + FEC + ring + CCU + module). */
  FedChannelConnection( uint16_t fec_crate, 
			uint16_t fec_slot, 
			uint16_t fec_ring, 
			uint16_t ccu_addr, 
			uint16_t ccu_chan, 
			uint16_t apv0 = 0,
			uint16_t apv1 = 0,
			uint32_t dcu_id = 0,
			uint32_t det_id = 0,
			uint16_t pairs  = 0,
			uint16_t fed_id = 0,
			uint16_t fed_ch = 0,
			uint16_t length = 0,
			bool dcu = false,
			bool pll = false,
			bool mux = false,
			bool lld = false )
    : fecCrate_(fec_crate), fecSlot_(fec_slot), fecRing_(fec_ring), ccuAddr_(ccu_addr), ccuChan_(ccu_chan),
    apv0_(apv0), apv1_(apv1),
    dcuId_(dcu_id), detId_(det_id), nApvPairs_(pairs), 
    fedId_(fed_id), fedCh_(fed_ch), length_(length),
    dcu0x00_(dcu), mux0x43_(mux), pll0x44_(pll), lld0x60_(lld) { dcu_id ? dcu0x00_=true : dcu0x00_=false; }
  
  /** Default constructor. */
  ~FedChannelConnection() {;}

  // ----- Control structure -----
  
  inline const uint16_t& fecCrate() const;
  inline const uint16_t& fecSlot() const;
  inline const uint16_t& fecRing() const;
  inline const uint16_t& ccuAddr() const;
  inline const uint16_t& ccuChan() const;

  // ----- APV devices -----

  /** Returns I2C address of APV0 or APV1. */
  const uint16_t& i2cAddr( const uint16_t& apv ) const; 
  
  // ----- Other hybrid devices -----

  inline const bool& dcu() const;
  inline const bool& mux() const;
  inline const bool& pll() const;
  inline const bool& lld() const;
  
  // ----- Module information -----

  /** Returns DCU id for associated module. */
  inline const uint32_t& dcuId() const;

  /** Returns LLD channel on hybrid (0->2) for this connection. */
  uint16_t lldChannel() const; 
  
  // ----- Detector information -----

  inline const uint32_t& detId() const;

  /** Returns APV pair number for this connection (this can be either
      0->1 or 0->2, depending on number of detector strips). */
  uint16_t apvPairNumber() const;

  /** Number of APV pairs for associated module. */
  inline const uint16_t& nApvPairs() const;
  
  /** Number of detector strips for associated module. */
  inline uint16_t nDetStrips() const;
  
  // ----- FED connection information -----

  inline const uint16_t& fedId() const;
  inline const uint16_t& fedCh() const;

  inline void fedId( uint16_t& fed_id );
  inline void fedCh( uint16_t& fed_ch );
  
  // ----- Miscellaneous -----

  /** Prints all connection information. */
  void print( std::stringstream& ) const;

 private: 

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

  static const std::string logCategory_;
  
};

// ---------- inline methods ----------

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
const uint16_t& FedChannelConnection::nApvPairs() const { return nApvPairs_; }
uint16_t FedChannelConnection::nDetStrips() const { return 256*nApvPairs_; }

const uint16_t& FedChannelConnection::fedId() const { return fedId_; }
const uint16_t& FedChannelConnection::fedCh() const { return fedCh_; }

void FedChannelConnection::fedId( uint16_t& fed_id ) { fedId_ = fed_id; }
void FedChannelConnection::fedCh( uint16_t& fed_ch ) { fedCh_ = fed_ch; }

#endif // CondFormats_SiStripObjects_FedChannelConnection_H

