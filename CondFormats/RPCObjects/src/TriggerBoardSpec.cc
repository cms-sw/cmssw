#include "CondFormats/RPCObjects/interface/TriggerBoardSpec.h"
#include <sstream>
#include <iostream>

typedef std::vector<LinkConnSpec>::const_iterator IT;

TriggerBoardSpec::TriggerBoardSpec(int num, uint32_t aMask) : theNum(num), theMaskedLinks(aMask) {}

const LinkConnSpec* TriggerBoardSpec::linkConn(int tbInputNumber) const {
  //FIXME - temporary implementaion, to be replace by LUT (in preparation)
  for (IT it = theLinks.begin(); it != theLinks.end(); it++) {
    if (tbInputNumber == it->triggerBoardInputNumber())
      return &(*it);
  }
  return nullptr;
}

std::vector<const LinkConnSpec*> TriggerBoardSpec::enabledLinkConns() const {
  std::vector<const LinkConnSpec*> result;
  for (IT it = theLinks.begin(); it != theLinks.end(); it++) {
    //
    // check that link is not masked!
    // std::cout <<"masked links:"<<theMaskedLinks<<std::endl;
    //
    result.push_back(&(*it));
  }
  return result;
}

std::string TriggerBoardSpec::print(int depth) const {
  std::ostringstream str;
  str << "TriggerBoardSpec: num=" << dccInputChannelNum() << std::endl;
  depth--;
  if (depth >= 0) {
    for (IT ic = theLinks.begin(); ic != theLinks.end(); ic++)
      str << (*ic).print(depth);
  }
  return str.str();
}

void TriggerBoardSpec::add(const LinkConnSpec& lc) { theLinks.push_back(lc); }
