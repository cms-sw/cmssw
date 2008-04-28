#include "RecoParticleFlow/PFClusterTools/interface/PFToolsException.hh"


using namespace pftools;

PFToolsException::PFToolsException(const std::string& aErrorDescription)
{
	myDescription = aErrorDescription;
}

PFToolsException::~PFToolsException() throw() 
{
}

const char* PFToolsException::what() const throw(){
	return myDescription.c_str();
}
