#include "CondFormats/RPCObjects/interface/RPCFebConnector.h"

#include <ostream>
#include <sstream>

RPCFebConnector::RPCFebConnector(RPCDetId const& rpc_det_id, unsigned int first_strip, int slope, std::uint16_t channels)
    : first_strip_(1), slope_(slope < 0 ? -1 : 1), channels_(channels), rpc_det_id_(rpc_det_id.rawId()) {
  setFirstStrip(first_strip);
}

std::string RPCFebConnector::getString() const {
  std::ostringstream oss;
  oss << rpc_det_id_ << '_' << (int)first_strip_ << (slope_ < 0 ? '-' : '+') << '_' << std::hex << std::showbase
      << channels_;
  return oss.str();
}

std::ostream& operator<<(std::ostream& ostream, RPCFebConnector const& connector) {
  return (ostream << connector.getString());
}
