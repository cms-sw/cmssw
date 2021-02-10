#ifndef SSTRING_H
#define SSTRING_H

#include "platformDependantOptions.h"
#include "DataBlock.h"
#include "Array.h"
#include <string>


#ifdef WIN32
typedef __int64		longlong;		// for millisecond timestamps
#else
#include <sys/types.h> 
typedef	long long int 	longlong;
#endif



/**
* std::string was not behaving itself under MSVC++ 6, so I wrote this
*/
class PLATFORMDEPENDANT_DLL_API SString{
private:

	DataBlock stringStore;

	void assign(const char * str);

public:
	static char null;

	SString();
	SString(const std::string &str);	
	SString(const SString & str);
	SString(const char * str);
	SString(const char * string, unsigned stringSize);

	
	virtual ~SString();


	SString & operator=(const std::string &str);
	SString & operator=(const SString & str);
	SString & operator=(const char * str);



	bool operator==(const std::string &str) const;
	bool operator==(const SString &str)const;
	bool operator==(const char *str)const;




	bool operator!=(const std::string &str) const;
	bool operator!=(const SString &str) const;
	bool operator!=(const char *str) const;




	
	SString & append(const char * str);
	SString & operator+=(const std::string &str);
	SString & operator+=(const SString &str);
	SString & operator+=(const char *str);


	SString operator+(const std::string &str) const;
	SString operator+(const SString &str) const;
	SString operator+(const char *str)const;
	SString operator+(const int val)const;
	SString operator+(const longlong val)const;
	SString operator+(const float val)const;
	SString operator+(const double val)const;


	SString& operator<<(const std::string &str);
	SString& operator<<(const SString &str);
	SString& operator<<(const char *str);
	SString& operator<<(const int val);
	SString& operator<<(const longlong val);
	SString& operator<<(const float val);
	SString& operator<<(const double val);


	char * getString() const;
	operator char*() const;
	operator const char*() const;
	operator const std::string() const;

	/**
	* not incluing null terminator
	*/
	unsigned int strLen() const;


	/**
	* will return a array of string separated at c.
	* SArray is owned by caller.
	* empty substrings will NOT be include. 
	*/
	SArray<SString> * splitAt(char c);
};

#endif
