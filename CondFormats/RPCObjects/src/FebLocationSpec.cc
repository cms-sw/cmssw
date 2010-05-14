#include "CondFormats/RPCObjects/interface/FebLocationSpec.h"
#include <sstream>

std::string FebLocationSpec::print(int depth) const
{
  std::ostringstream str;
  std::string localPartVal[6]={"Forward","Central","Backward","A","B","C"};
  std::string cmsPartVal[6]={"1","2","3","A","B","C"};
  if(depth >= 0) {
  str << "FebLocationSpec: " << std::endl
            <<" local partition: "<<localPartVal[localEtaPartition-1]<<" ("<<int(positionInLocalEtaPartition) <<")"
            <<" cms partition: " <<cmsPartVal[cmsEtaPartition-1] <<" ("<<int(positionInCmsEtaPartition)<<")"
            << std::endl;
  }
  return str.str();
}
