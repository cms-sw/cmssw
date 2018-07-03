#ifndef PFTOOLSEXCEPTION_H_
#define PFTOOLSEXCEPTION_H_

#include <exception>
#include <string>
namespace pftools {
/**
 \class PFToolsException 
 \brief General purpose exception class for use by classes in the pftools namespace

 \author Jamie Ballin
 \date   April 2008
 */
class PFToolsException : public std::exception {
public:
	PFToolsException(const std::string& aErrorDescription="");
	
	~PFToolsException() noexcept override;
	
	const char* what() const noexcept override;
	
protected:
	std::string myDescription;
};
}

#endif /*PFTOOLSEXCEPTION_HH_*/
