#ifndef CondFormats_SiStripObjects_FedChannelConnection_H
#define CondFormats_SiStripObjects_FedChannelConnection_H

#include <boost/cstdint.hpp>
#include <vector>

/** 
    \class FedChannelConnection 
    \brief Channel-level device and connection information.
*/
class FedChannelConnection {
  
 public:
  
  FedChannelConnection()
    : fecCrate_(0), fecSlot_(0), fecRing_(0), ccuAddr_(0), ccuChan_(0),
    apv0_(0), apv1_(0), dcu0x00_(false), mux0x43_(false), pll0x44_(false), lld0x60_(false),
    dcuId_(0), detId_(0), nPairs_(0),
    fedId_(0), fedCh_(0), length_(0) {;}
  
  FedChannelConnection( unsigned short fec_crate, 
			unsigned short fec_slot, 
			unsigned short fec_ring, 
			unsigned short ccu_addr, 
			unsigned short ccu_chan, 
			unsigned short apv0 = 0,
			unsigned short apv1 = 0,
			bool dcu = false,
			bool pll = false,
			bool mux = false,
			bool lld = false,
			unsigned int   dcu_id = 0,
			uint32_t       det_id = 0,
			unsigned short pairs  = 0,
			unsigned short fed_id = 0,
			unsigned short fed_ch = 0,
			unsigned short length = 0 )
    : fecCrate_(fec_crate), fecSlot_(fec_slot), fecRing_(fec_ring), ccuAddr_(ccu_addr), ccuChan_(ccu_chan),
    apv0_(apv0), apv1_(apv1), dcu0x00_(dcu), mux0x43_(mux), pll0x44_(pll), lld0x60_(lld),
    dcuId_(dcu_id), detId_(det_id), nPairs_(pairs), 
    fedId_(fed_id), fedCh_(fed_ch), length_(length) {;}
  
  ~FedChannelConnection() {;}

  // Control   
  const unsigned short& fecCrate() const { return fecCrate_; } 
  const unsigned short& fecSlot()  const { return fecSlot_; } 
  const unsigned short& fecRing()  const { return fecRing_; }
  const unsigned short& ccuAddr()  const { return ccuAddr_; }
  const unsigned short& ccuChan()  const { return ccuChan_; }

  // I2C addresses
  const unsigned short& i2cAddrApv0() const { return apv0_; }
  const unsigned short& i2cAddrApv1() const { return apv1_; }

  const bool& dcu() const { return dcu0x00_; }
  const bool& mux() const { return mux0x43_; }
  const bool& pll() const { return pll0x44_; }
  const bool& lld() const { return lld0x60_; }
  
  // Module / Detector
  const unsigned int&   dcuId()  const { return dcuId_; }
  const uint32_t&       detId()  const { return detId_; }
  const unsigned short& nPairs() const { return nPairs_; }
  unsigned short pairPos() const;
  unsigned short pairId()  const;

  // FED
  const unsigned short& fedId() const { return fedId_; }
  const unsigned short& fedCh() const { return fedCh_; }

 private: 

  // Control
  unsigned short fecCrate_;
  unsigned short fecSlot_;
  unsigned short fecRing_;
  unsigned short ccuAddr_;
  unsigned short ccuChan_;

  // I2C addresses
  unsigned short apv0_; 
  unsigned short apv1_; 

  // Found devices
  bool dcu0x00_; 
  bool mux0x43_; 
  bool pll0x44_; 
  bool lld0x60_; 

  // Module / Detector
  unsigned int   dcuId_;
  uint32_t       detId_;
  unsigned short nPairs_;

  // FED
  unsigned short fedId_;
  unsigned short fedCh_;
  unsigned short length_;
  
};

#endif // CondFormats_SiStripObjects_FedChannelConnection_H

