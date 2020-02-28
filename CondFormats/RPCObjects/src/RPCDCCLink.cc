#include "CondFormats/RPCObjects/interface/RPCDCCLink.h"

#include <ostream>
#include <sstream>

RPCDCCLink::RPCDCCLink() : id_(0x0) {}

RPCDCCLink::RPCDCCLink(std::uint32_t const& id) : id_(id) {}

RPCDCCLink::RPCDCCLink(int fed, int dccinput, int tbinput) : id_(0x0) {
  setFED(fed);
  setDCCInput(dccinput);
  setTBInput(tbinput);
}

std::uint32_t RPCDCCLink::getMask() const {
  std::uint32_t mask(0x0);
  if (id_ & mask_fed_)
    mask |= mask_fed_;
  if (id_ & mask_dccinput_)
    mask |= mask_dccinput_;
  if (id_ & mask_tbinput_)
    mask |= mask_tbinput_;
  return mask;
}

std::string RPCDCCLink::getName() const {
  std::ostringstream oss;
  oss << "RPCDCCLink_";
  bf_stream(oss, min_fed_, mask_fed_, pos_fed_);
  bf_stream(oss << '_', min_dccinput_, mask_dccinput_, pos_dccinput_);
  bf_stream(oss << '_', min_tbinput_, mask_tbinput_, pos_tbinput_);
  return oss.str();
}

std::ostream& operator<<(std::ostream& ostream, RPCDCCLink const& link) { return (ostream << link.getName()); }
