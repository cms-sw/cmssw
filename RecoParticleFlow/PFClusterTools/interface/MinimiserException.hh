#ifndef MINIMISEREXCEPTION_H_
#define MINIMISEREXCEPTION_H_

#include <exception>
#include <string>
namespace minimiser {
class MinimiserException : public std::exception {
public:
	MinimiserException(const std::string& aErrorDescription="");
	
	virtual ~MinimiserException() throw();
	
	virtual const char* what() const throw();
	
protected:
	std::string myDescription;
};
}

#endif /*MINIMISEREXCEPTION_HH_*/
