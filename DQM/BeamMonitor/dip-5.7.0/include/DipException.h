#ifndef DIPEXCEPTION_H_INCLUDED
#define DIPEXCEPTION_H_INCLUDED
#include "Options.h"
#include <string>
#include <exception>

#ifndef WIN32
#include <stdexcept>
/*class std::runtime_error{
 private:
  std::string why;
 public:
  runtime_error(const std::string & reason):why(reason){
  }

  virtual ~runtime_error(){};

  virtual const char* what () const{
    return why.c_str();
  }
};*/
#endif

/**
* Base class for DIP exceptions
*/
class DipDllExp DipException:public std::runtime_error{
public:
	DipException(const std::string &msg);

	virtual ~DipException() throw (){};
};




/**
* Thrown when there is an error within DIP
*/
class DipDllExp DipInternalError: public DipException
{
public:
	DipInternalError(const std::string & message);

	~DipInternalError()  throw (){};
};




/**
* Used when throwing errors which are specific to the
* DIM implementation of DIP
*/
class DipDllExp DipDIMInternalError: public DipInternalError
{
private:
	/// The error codes (defined in dim_common.h)
	const int errorCode;

public:
	/// Set the error information
	DipDIMInternalError(const std::string & message, int theErrorCode);

	~DipDIMInternalError() throw (){};
	
	/// Retrieve error code.
	int getErrorCode();
};





/**
* Thrown when a type requested during a method call is not the actual type of the
* object requested.
*/
class DipDllExp TypeMismatch: public DipException
{
public:
	TypeMismatch(std::string & msg);


	TypeMismatch();

	~TypeMismatch() throw (){};
};



/**
* Usually thrown when parameter pass in method call has a bad value
*/
class DipDllExp BadParameter:public DipException
{

public:
	BadParameter(std::string & msg);

	BadParameter();

	~BadParameter() throw (){};
};



//std::ostream& operator<<(std::ostream& theStream, DipException& theException);

#endif //DIPEXCEPTION_H_INCLUDED

