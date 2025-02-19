#ifndef PhysicsTools_MVATrainer_XMLSimpleStr_h
#define PhysicsTools_MVATrainer_XMLSimpleStr_h

#include <string>

#include <xercesc/util/XMLString.hpp>
#include <xercesc/util/XMLUni.hpp>

class XMLSimpleStr {
    public:
	XMLSimpleStr(const XMLCh *str) :
		string(XERCES_CPP_NAMESPACE_QUALIFIER XMLString::transcode(str))
	{}

	~XMLSimpleStr()
	{ XERCES_CPP_NAMESPACE_QUALIFIER XMLString::release(&string); }

	operator const char *() const
	{ return string; }

	operator std::string() const
	{ return string; }

    private:
	char	*string;
};

#endif // PhysicsTools_MVATrainer_XMLSimpleStr_h
