#include "CondFormats/RPCObjects/interface/FebLocationSpec.h"
#include <sstream>

std::string FebLocationSpec::localEtaPartitionName() const {
  std::ostringstream str;
  const static std::string localPartVal[7] = {"Forward", "Central", "Backward", "A", "B", "C", "D"};
  str << localPartVal[localEtaPartition - 1];
  return str.str();
}

std::string FebLocationSpec::print(int depth) const {
  std::ostringstream str;
  std::string cmsPartVal[6] = {"1", "2", "3", "A", "B", "C"};
  if (depth >= 0) {
    str << "FebLocationSpec: " << std::endl
        << " local partition: " << localEtaPartitionName() << " (" << int(positionInLocalEtaPartition) << ")"
        << " cms partition: " << cmsPartVal[cmsEtaPartition - 1] << " (" << int(positionInCmsEtaPartition) << ")"
        << std::endl;
  }
  return str.str();
}
