#include "PhysicsTools/CommonTools/interface/InputTagDistributor.h"

InputTagDistributor* InputTagDistributor::SetInputTagDistributorUniqueInstance_=0;
std::map<std::string, InputTagDistributor*> InputTagDistributor::multipleInstance_=std::map<std::string, InputTagDistributor*>();

