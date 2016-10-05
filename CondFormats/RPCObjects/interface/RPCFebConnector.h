#ifndef CondFormats_RPCObjects_RPCFebConnector_h
#define CondFormats_RPCObjects_RPCFebConnector_h

#include <stdint.h>
#include <string>
#include <iosfwd>

#include "CondFormats/Serialization/interface/Serializable.h"

#include "DataFormats/MuonDetId/interface/RPCDetId.h"

class RPCFebConnector
{
public:
    static unsigned int const min_first_strip_ = 1;
    static unsigned int const max_first_strip_ = 128;
    static unsigned int const nchannels_       = 16;

    static unsigned int bit_count(::uint16_t);

public:
    RPCFebConnector(RPCDetId const & _rpc_det_id = RPCDetId(0, 0, 1, 1, 1, 1, 0)
                    , unsigned int _first_strip = 1
                    , int _slope = 1
                    , ::uint16_t _channels = 0x0);

    void reset();

    RPCDetId getRPCDetId() const;
    unsigned int getFirstStrip() const;
    int getSlope() const;
    ::uint16_t getChannels() const;

    RPCFebConnector & setRPCDetId(RPCDetId const & _rpc_det_id);
    RPCFebConnector & setFirstStrip(unsigned int _strip);
    RPCFebConnector & setSlope(int _slope);
    RPCFebConnector & setChannels(::uint16_t _channels);

    bool isActive(unsigned int _channel) const;
    unsigned int getNChannels() const;
    unsigned int getStrip(unsigned int _channel) const;

    bool hasStrip(unsigned int _strip) const;
    unsigned int getChannel(unsigned int _strip) const;

    std::string getString() const;

protected:
    ::uint8_t  first_strip_; ///< strip, allowing range [1-128]
    ::int8_t   slope_;       ///< -1 or 1
    ::uint16_t channels_;    ///< active channels in range [1-16]

    ::uint32_t rpc_det_id_;

    COND_SERIALIZABLE;
};

std::ostream & operator<<(std::ostream & _ostream, RPCFebConnector const & _connector);

#include "CondFormats/RPCObjects/interface/RPCFebConnector.icc"

#endif // CondFormats_RPCObjects_RPCFebConnector_h
