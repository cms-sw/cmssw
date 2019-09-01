#ifndef CondFormats_RPCObjects_RPCFebConnector_h
#define CondFormats_RPCObjects_RPCFebConnector_h

#include <cstdint>
#include <string>
#include <iosfwd>

#include "CondFormats/Serialization/interface/Serializable.h"

#include "DataFormats/MuonDetId/interface/RPCDetId.h"

class RPCFebConnector {
public:
  static unsigned int const min_first_strip_ = 1;
  static unsigned int const max_first_strip_ = 128;
  static unsigned int const nchannels_ = 16;

  static unsigned int bit_count(std::uint16_t);

public:
  RPCFebConnector(RPCDetId const& rpc_det_id = RPCDetId(0, 0, 1, 1, 1, 1, 0),
                  unsigned int first_strip = 1,
                  int slope = 1,
                  std::uint16_t channels = 0x0);

  void reset();

  RPCDetId getRPCDetId() const;
  unsigned int getFirstStrip() const;
  int getSlope() const;
  std::uint16_t getChannels() const;

  RPCFebConnector& setRPCDetId(RPCDetId const& rpc_det_id);
  RPCFebConnector& setFirstStrip(unsigned int strip);
  RPCFebConnector& setSlope(int slope);
  RPCFebConnector& setChannels(std::uint16_t channels);

  bool isActive(unsigned int channel) const;
  unsigned int getNChannels() const;
  unsigned int getStrip(unsigned int channel) const;

  bool hasStrip(unsigned int strip) const;
  unsigned int getChannel(unsigned int strip) const;

  std::string getString() const;

protected:
  std::uint8_t first_strip_;  ///< strip, allowing range [1-128]
  ::int8_t slope_;            ///< -1 or 1
  std::uint16_t channels_;    ///< active channels in range [1-16]

  std::uint32_t rpc_det_id_;

  COND_SERIALIZABLE;
};

std::ostream& operator<<(std::ostream& ostream, RPCFebConnector const& connector);

#include "CondFormats/RPCObjects/interface/RPCFebConnector.icc"

#endif  // CondFormats_RPCObjects_RPCFebConnector_h
