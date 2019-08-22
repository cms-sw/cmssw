#include "CondFormats/RPCObjects/interface/RPCLBLink.h"

#include <ostream>
#include <sstream>

RPCLBLink::RPCLBLink() : id_(0x0) {}

RPCLBLink::RPCLBLink(std::uint32_t const& id) : id_(id) {}

RPCLBLink::RPCLBLink(
    int region, int yoke, int sector, int side, int wheelordisk, int fibre, int radial, int linkboard, int connector)
    : id_(0x0) {
  setRegion(region);
  setYoke(yoke);
  setSector(sector);
  setSide(side);
  setWheelOrDisk(wheelordisk);
  setFibre(fibre);
  setRadial(radial);
  setLinkBoard(linkboard);
  setConnector(connector);
}

std::uint32_t RPCLBLink::getMask() const {
  std::uint32_t mask(0x0);
  if (id_ & mask_region_)
    mask |= mask_region_;
  if (id_ & mask_yoke_)
    mask |= mask_yoke_;
  if (id_ & mask_sector_)
    mask |= mask_sector_;
  if (id_ & mask_side_)
    mask |= mask_side_;
  if (id_ & mask_wheelordisk_)
    mask |= mask_wheelordisk_;
  if (id_ & mask_fibre_)
    mask |= mask_fibre_;
  if (id_ & mask_radial_)
    mask |= mask_radial_;
  if (id_ & mask_linkboard_)
    mask |= mask_linkboard_;
  if (id_ & mask_connector_)
    mask |= mask_connector_;
  return mask;
}

std::string RPCLBLink::getName() const {
  // LB_Rregion.yoke_Ssector_region.side.wheel_or_disk.fibre.radial_CHlinkboard:connector
  // LB_RB      -2  _S10    _ B     N    2             A           _CH2        :
  // LB_RE      -1  _S10    _ E     N    2             3           _CH0        :
  // LB_RB-2_S10_BN2A_CH2 , RB1in/W-2/S10:bwd ; LB_RE-1_S10_EN23_CH0 , RE-2/R3/C30

  int region(getRegion()), yoke(getYoke()), linkboard(getLinkBoard()), connector(getConnector());

  std::ostringstream oss;
  oss << "LB_R";
  switch (region) {
    case 0:
      oss << 'B';
      break;
    case 1:
      oss << 'E';
      break;
    default:
      oss << '*';
      break;
  }
  (yoke > 0 ? oss << '+' << yoke : oss << yoke);

  bf_stream(oss << "_S", min_sector_, mask_sector_, pos_sector_);

  oss << '_';
  switch (region) {
    case 0:
      oss << 'B';
      break;
    case 1:
      oss << 'E';
      break;
    default:
      oss << '*';
      break;
  }
  switch (getSide()) {
    case 0:
      oss << 'N';
      break;
    case 1:
      oss << 'M';
      break;
    case 2:
      oss << 'P';
      break;
    default:
      oss << '*';
      break;
  }
  bf_stream(oss, min_wheelordisk_, mask_wheelordisk_, pos_wheelordisk_);
  switch (getFibre()) {
    case 0:
      oss << '1';
      break;
    case 1:
      oss << '2';
      break;
    case 2:
      oss << '3';
      break;
    case 3:
      oss << 'A';
      break;
    case 4:
      oss << 'B';
      break;
    case 5:
      oss << 'C';
      break;
    case 6:
      oss << 'D';
      break;
    case 7:
      oss << 'E';
      break;
    default:
      oss << '*';
      break;
  }
  switch (getRadial()) {  // for completeness, CMS IN 2002/065
    case 0:
      oss << "ab";
      break;
    case 1:
      oss << "cd";
      break;
    default:
      oss << "";
      break;
  }

  if (linkboard != wildcard_)
    oss << "_CH" << linkboard;

  if (connector != wildcard_)
    oss << ":" << connector;

  return oss.str();
}

std::ostream& operator<<(std::ostream& ostream, RPCLBLink const& link) { return (ostream << link.getName()); }
