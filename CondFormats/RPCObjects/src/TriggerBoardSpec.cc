#include "CondFormats/RPCObjects/interface/TriggerBoardSpec.h"
#include <sstream>
#include <iostream>

typedef std::vector<LinkConnSpec>::const_iterator IT;

TriggerBoardSpec::TriggerBoardSpec(int num, uint32_t aMask) : theNum(num), theMaskedLinks(aMask) {}

const LinkConnSpec* TriggerBoardSpec::linkConn(int tbInputNumber) const {
  //FIXME - temporary implementaion, to be replace by LUT (in preparation)
  for (const auto& theLink : theLinks) {
    if (tbInputNumber == theLink.triggerBoardInputNumber())
      return &theLink;
  }
  return nullptr;
}

std::vector<const LinkConnSpec*> TriggerBoardSpec::enabledLinkConns() const {
  std::vector<const LinkConnSpec*> result;
  for (const auto& theLink : theLinks) {
    //
    // check that link is not masked!
    // std::cout <<"masked links:"<<theMaskedLinks<<std::endl;
    //
    result.push_back(&theLink);
  }
  return result;
}

std::string TriggerBoardSpec::print(int depth) const {
  std::ostringstream str;
  str << "TriggerBoardSpec: num=" << dccInputChannelNum() << std::endl;
  depth--;
  if (depth >= 0) {
    for (const auto& theLink : theLinks)
      str << theLink.print(depth);
  }
  return str.str();
}

void TriggerBoardSpec::add(const LinkConnSpec& lc) { theLinks.push_back(lc); }
