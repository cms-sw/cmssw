#include "CondFormats/RPCObjects/interface/FebLocationSpec.h"
#include <sstream>

std::string FebLocationSpec::print(int depth) const
{
  std::ostringstream str;
  if(depth >= 0) {
  str << "FebLocationSpec: " << std::endl
            <<" local partition: "<<localEtaPartition<<" ("<<positionInLocalEtaPartition <<")"
            <<" cms partition: " <<cmsEtaPartition <<" ("<<positionInCmsEtaPartition<<")"
            << std::endl;
  }
  return str.str();
}
