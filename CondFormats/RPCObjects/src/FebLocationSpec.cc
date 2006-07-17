#include "CondFormats/RPCObjects/interface/FebLocationSpec.h"
#include <iostream>

void FebLocationSpec::print(int depth) const
{
  if(depth<0) return;
  std::cout << "FebLocationSpec: " << std::endl
            <<" local partition: " 
              <<positionInCmsEtaPartition
              <<" ("<<localEtaPartition<<")"
            <<" cms partition: "
              <<cmsEtaPartition
              <<" ("<<positionInCmsEtaPartition<<")"
            << std::endl;
}
