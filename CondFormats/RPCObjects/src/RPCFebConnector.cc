#include "CondFormats/RPCObjects/interface/RPCFebConnector.h"

#include <ostream>
#include <sstream>

RPCFebConnector::RPCFebConnector(RPCDetId const & _rpc_det_id
                                 , unsigned int _first_strip
                                 , int _slope
                                 , ::uint16_t _channels)
    : first_strip_(1)
    , slope_(_slope < 0 ? -1 : 1)
    , channels_(_channels)
    , rpc_det_id_(_rpc_det_id.rawId())
{
    setFirstStrip(_first_strip);
}

std::string RPCFebConnector::getString() const
{
    std::ostringstream _oss;
    _oss << rpc_det_id_ << '_'
         << (int)first_strip_ << (slope_ < 0 ? '-' : '+') << '_'
         << std::hex << std::showbase << channels_;
    return _oss.str();
}

std::ostream & operator<<(std::ostream & _ostream, RPCFebConnector const & _connector)
{
    return (_ostream << _connector.getString());
}
