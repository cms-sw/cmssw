#include "CondFormats/RPCObjects/interface/LinkBoardElectronicIndex.h"
#include <sstream>

using namespace std;

std::string LinkBoardElectronicIndex::print(int depth) const {
  ostringstream str;
  if (depth >= 0)
    str << " ---> dccId: " << dccId << " dccInputChannelNum: " << dccInputChannelNum
        << " tbLinkInputNum: " << tbLinkInputNum << " lbNumInLink: " << lbNumInLink;

  return str.str();
}
