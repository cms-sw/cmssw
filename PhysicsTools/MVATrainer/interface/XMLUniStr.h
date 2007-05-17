#ifndef PhysicsTools_MVATrainer_XMLUniStr_h
#define PhysicsTools_MVATrainer_XMLUniStr_h

#include <xercesc/util/XMLString.hpp>
#include <xercesc/util/XMLUni.hpp>

class XMLUniStr {
    public:
	XMLUniStr(const char *str) :
		unicode(XERCES_CPP_NAMESPACE_QUALIFIER XMLString::transcode(str))
	{}

	~XMLUniStr()
	{ XERCES_CPP_NAMESPACE_QUALIFIER XMLString::release(&unicode); }

	operator const XMLCh *() const
	{ return unicode; }

    private:
	XMLCh	*unicode;
};

#endif // PhysicsTools_MVATrainer_XMLUniStr_h
