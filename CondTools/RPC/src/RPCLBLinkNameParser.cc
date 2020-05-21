#include "CondTools/RPC/interface/RPCLBLinkNameParser.h"

#include <sstream>

#include "FWCore/Utilities/interface/Exception.h"

void RPCLBLinkNameParser::parse(std::string const& name, RPCLBLink& lb_link) {
  lb_link.reset();
  std::string::size_type size = name.size();
  std::string::size_type pos(0), next(0);
  int tmp;

  std::istringstream conv;

  // region
  pos = name.find("_R", pos);
  if (pos == std::string::npos || (pos += 2) >= size)
    throw cms::Exception("InvalidLinkBoardName") << "Expected _R[region], got " << name;
  switch (name.at(pos)) {
    case 'B':
      lb_link.setRegion(0);
      break;
    case 'E':
      lb_link.setRegion(1);
      break;
    default:
      throw cms::Exception("InvalidLinkBoardName") << "Expected Region B or E, got " << name.at(pos) << " in " << name;
      break;
  }
  if ((++pos) >= size)
    throw cms::Exception("InvalidLinkBoardName") << "Name too short: " << name;

  // yoke
  next = name.find_first_not_of("+-0123456789", pos);
  conv.clear();
  conv.str(name.substr(pos, next - pos));
  conv >> tmp;
  lb_link.setYoke(tmp);
  pos = next;

  // sector
  pos = name.find("_S", pos);
  if (pos == std::string::npos || (pos += 2) >= size)
    throw cms::Exception("InvalidLinkBoardName") << "Expected _S[sector], got " << name;
  next = name.find_first_not_of("+-0123456789", pos);
  conv.clear();
  conv.str(name.substr(pos, next - pos));
  conv >> tmp;
  lb_link.setSector(tmp);
  pos = next;

  // (region) side
  pos = name.find('_', pos);
  if (pos == std::string::npos || (pos += 2) >= size)
    throw cms::Exception("InvalidLinkBoardName") << "Name too short: " << name;
  switch (name.at(pos)) {
    case 'N':
      lb_link.setSide(0);
      break;
    case 'M':
      lb_link.setSide(1);
      break;
    case 'P':
      lb_link.setSide(2);
      break;
    default:
      throw cms::Exception("InvalidLinkBoardName") << "Expected Side N, M or P, got " << name.at(pos) << " in " << name;
      break;
  }
  if ((++pos) >= size)
    throw cms::Exception("InvalidLinkBoardName") << "Name too short: " << name;

  // wheelordisk
  conv.clear();
  conv.str(name.substr(pos, 1));
  conv >> tmp;
  lb_link.setWheelOrDisk(tmp);
  if ((++pos) >= size)
    throw cms::Exception("InvalidLinkBoardName") << "Name too short: " << name;

  // fibre
  {
    std::string fibre("123ABCDE");
    char const* tmpchar = std::find(&(fibre[0]), &(fibre[0]) + 8, name.at(pos));
    lb_link.setFibre(tmpchar - &(fibre[0]));
  }
  if ((++pos) >= size)
    return;

  // radial
  next = name.find("_CH", pos);
  if (next == std::string::npos)
    next = size;
  if (next - pos == 2) {
    std::string radial = name.substr(pos, 2);
    if (radial == "ab")
      lb_link.setRadial(0);
    else if (radial == "cd")
      lb_link.setRadial(1);
  }

  if (next == size)
    return;

  // linkboard
  pos = next;
  if (pos + 3 >= size)
    throw cms::Exception("InvalidLinkBoardName") << "Name too short: " << name;
  pos += 3;
  next = name.find_first_not_of("+-0123456789", pos);
  conv.clear();
  conv.str(name.substr(pos, next - pos));
  conv >> tmp;
  lb_link.setLinkBoard(tmp);
}

RPCLBLink RPCLBLinkNameParser::parse(std::string const& name) {
  RPCLBLink lb_link;
  parse(name, lb_link);
  return lb_link;
}
