#include <iostream>
#include <memory>
#include <string>

#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/util/XMLUni.hpp>
#include <xercesc/sax2/SAX2XMLReader.hpp>
#include <xercesc/sax2/XMLReaderFactory.hpp>

#include "FWCore/Utilities/interface/Exception.h"

#include "XMLUtils.h"

XERCES_CPP_NAMESPACE_USE

namespace lhef {

unsigned int XMLDocument::XercesPlatform::instances = 0;

XMLDocument::XercesPlatform::XercesPlatform()
{
	if (!instances++) {
		try {
			XMLPlatformUtils::Initialize();
		} catch(const XMLException &e) {
			throw cms::Exception("XMLDocument")
				<< "XMLPlatformUtils::Initialize failed "
				   "because of: "
				<< XMLSimpleStr(e.getMessage()) << std::endl;
		}
	}
}

XMLDocument::XercesPlatform::~XercesPlatform()
{
	if (!--instances)
		XMLPlatformUtils::Terminate();
}

XMLDocument::XMLDocument(std::auto_ptr<std::istream> &in, Handler &handler) :
	platform(new XercesPlatform()),
	source(new STLInputSource(in)),
	parser(XMLReaderFactory::createXMLReader()),
	done(false)
{
	parser->setFeature(XMLUni::fgSAX2CoreValidation, false);
	parser->setFeature(XMLUni::fgSAX2CoreNameSpaces, false);
	parser->setFeature(XMLUni::fgXercesSchema, false);
	parser->setFeature(XMLUni::fgXercesSchemaFullChecking, false);

	parser->setContentHandler(&handler);
	parser->setLexicalHandler(&handler);
	parser->setErrorHandler(&handler);

	if (!parser->parseFirst(*source, token))
		throw cms::Exception("XMLParseError")
			<< "SAXParser::parseFirst failed" << std::endl;
}

XMLDocument::~XMLDocument()
{
}

bool XMLDocument::parse()
{
	if (done || parser->getErrorCount())
		return false;

	done = !parser->parseNext(token);

	return !done;
}

STLInputStream::STLInputStream(std::istream &in) :
	in(in)
{
	if (in.bad())
		throw cms::Exception("FileStreamError")
			<< "I/O stream bad in STLInputStream::STLInputStream()"
			<< std::endl;
}

STLInputStream::~STLInputStream()
{
}

unsigned int STLInputStream::readBytes(XMLByte* const buf,
                                       const unsigned int size)
{
	char *rawBuf = reinterpret_cast<char*>(buf);
	unsigned int bytes = size * sizeof(XMLByte);
	in.read(rawBuf, bytes);
	unsigned int readBytes = in.gcount();

	if (in.bad())
		throw cms::Exception("FileStreamError")
			<< "I/O stream bad in STLInputStream::readBytes()"
			<< std::endl;

	unsigned int read = (unsigned int)(readBytes / sizeof(XMLByte));
	unsigned int rest = (unsigned int)(readBytes % sizeof(XMLByte));
	for(unsigned int i = 1; i <= rest; i++)
		in.putback(rawBuf[readBytes - i]);

	pos += readBytes;

	return read;
}

STLInputSource::STLInputSource(std::auto_ptr<std::istream> &in) :
	in(in)
{
}

STLInputSource::~STLInputSource()
{
}

} // namespace lhef
