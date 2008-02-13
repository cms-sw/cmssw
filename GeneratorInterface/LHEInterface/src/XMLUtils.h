#ifndef GeneratorInterface_LHEInterface_XMLUtils_h
#define GeneratorInterface_LHEInterface_XMLUtils_h

#include <iostream>
#include <string>
#include <memory>

#include <xercesc/util/XMLString.hpp>
#include <xercesc/util/XMLUni.hpp>
#include <xercesc/util/XMLChar.hpp>
#include <xercesc/util/BinInputStream.hpp>
#include <xercesc/framework/XMLPScanToken.hpp>
#include <xercesc/sax/InputSource.hpp>
#include <xercesc/sax2/DefaultHandler.hpp>
#include <xercesc/sax2/SAX2XMLReader.hpp>

namespace lhef {

class XMLDocument {
    public:
	class Handler : public XERCES_CPP_NAMESPACE_QUALIFIER DefaultHandler {};

	XMLDocument(std::auto_ptr<std::istream> &in, Handler &handler);
	virtual ~XMLDocument();

	bool parse();

    private:
	class XercesPlatform {
	    public:
		XercesPlatform();
		~XercesPlatform();

	    private:
		// do not make any kind of copies
		XercesPlatform(const XercesPlatform &orig);
		XercesPlatform &operator = (const XercesPlatform &orig);

		static unsigned int instances;
	};

	std::auto_ptr<XercesPlatform>					platform;

	std::auto_ptr<XERCES_CPP_NAMESPACE_QUALIFIER InputSource>	source;
	std::auto_ptr<XERCES_CPP_NAMESPACE_QUALIFIER SAX2XMLReader>	parser;

	XERCES_CPP_NAMESPACE_QUALIFIER XMLPScanToken			token;

	bool								done;
};

class XMLSimpleStr {
    public:
	XMLSimpleStr(const XMLCh *str) :
		string(XERCES_CPP_NAMESPACE_QUALIFIER XMLString::transcode(str))
	{}

	~XMLSimpleStr()
	{ XERCES_CPP_NAMESPACE_QUALIFIER XMLString::release(&string); }

	inline operator const char *() const
	{ return string; }

	inline static bool isAllSpaces(const XMLCh *str, unsigned int length)
	{ return XERCES_CPP_NAMESPACE_QUALIFIER
				XMLChar1_0::isAllSpaces(str, length); }

	inline static bool isSpace(XMLCh ch)
	{ return XERCES_CPP_NAMESPACE_QUALIFIER
				XMLChar1_0::isWhitespace(ch); }

    private:
	char	*string;
};

class STLInputStream : public XERCES_CPP_NAMESPACE_QUALIFIER BinInputStream {
    public:
	STLInputStream(std::istream &in);
	virtual ~STLInputStream();

	virtual unsigned int curPos() const { return pos; }

	virtual unsigned int readBytes(XMLByte *const buf,
	                               const unsigned int size);

    private:
	std::istream	&in;
	unsigned int	pos;
};

class STLInputSource : public XERCES_CPP_NAMESPACE_QUALIFIER InputSource {
    public:
	STLInputSource(std::auto_ptr<std::istream> &in);
	virtual ~STLInputSource();

	virtual XERCES_CPP_NAMESPACE_QUALIFIER BinInputStream* makeStream() const
	{ return new STLInputStream(*in); }

    private:
	std::auto_ptr<std::istream>	in;
};

} // namespace lhef

#endif // GeneratorInterface_LHEInterface_XMLUtils_h
