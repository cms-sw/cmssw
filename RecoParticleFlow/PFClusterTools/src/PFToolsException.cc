#include "RecoParticleFlow/PFClusterTools/interface/PFToolsException.h"


using namespace pftools;

PFToolsException::PFToolsException(const std::string& aErrorDescription)
{
	myDescription = aErrorDescription;
}

PFToolsException::~PFToolsException() noexcept
{
}

const char* PFToolsException::what() const noexcept {
	return myDescription.c_str();
}
