#include "CondFormats/RPCObjects/interface/RPCAMCLink.h"

#include <ostream>
#include <sstream>

RPCAMCLink::RPCAMCLink() : id_(0x0) {}

RPCAMCLink::RPCAMCLink(std::uint32_t const& id) : id_(id) {}

RPCAMCLink::RPCAMCLink(int fed, int amcnumber, int amcinput) : id_(0x0) {
  setFED(fed);
  setAMCNumber(amcnumber);
  setAMCInput(amcinput);
}

std::uint32_t RPCAMCLink::getMask() const {
  std::uint32_t mask(0x0);
  if (id_ & mask_fed_)
    mask |= mask_fed_;
  if (id_ & mask_amcnumber_)
    mask |= mask_amcnumber_;
  if (id_ & mask_amcinput_)
    mask |= mask_amcinput_;
  return mask;
}

std::string RPCAMCLink::getName() const {
  std::ostringstream oss;
  oss << "RPCAMCLink_";
  bf_stream(oss, min_fed_, mask_fed_, pos_fed_);
  if (id_ & (mask_amcnumber_ | mask_amcinput_)) {
    bf_stream(oss << '_', min_amcnumber_, mask_amcnumber_, pos_amcnumber_);
    if (id_ & mask_amcinput_) {
      bf_stream(oss << '_', min_amcinput_, mask_amcinput_, pos_amcinput_);
    }
  }
  return oss.str();
}

std::ostream& operator<<(std::ostream& ostream, RPCAMCLink const& link) { return (ostream << link.getName()); }
