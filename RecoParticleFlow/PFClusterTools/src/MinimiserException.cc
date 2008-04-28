#include "RecoParticleFlow/PFClusterTools/interface/MinimiserException.hh"


using namespace pftools;

MinimiserException::MinimiserException(const std::string& aErrorDescription)
{
	myDescription = aErrorDescription;
}

MinimiserException::~MinimiserException() throw() 
{
}

const char* MinimiserException::what() const throw(){
	return myDescription.c_str();
}
