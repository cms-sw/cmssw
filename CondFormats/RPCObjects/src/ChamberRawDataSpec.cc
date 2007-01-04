#include "CondFormats/RPCObjects/interface/ChamberRawDataSpec.h"
#include <sstream>

using namespace std;

std::string ChamberRawDataSpec::print( int depth ) const
{
  ostringstream str;
  if (depth >= 0) 
      str << " ---> dccId: " << dccId
          << " dccInputChannelNum: " << dccInputChannelNum
          << " tbLinkInputNum: " << tbLinkInputNum
          << " lbNumInLink: " << lbNumInLink;

  return str.str(); 
}
