#include <assert.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <memory>
#include <string>
#include <cstdio>
#include <stdio.h>
#include <ext/stdio_filebuf.h>

#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/util/XMLUni.hpp>
#include <xercesc/util/BinInputStream.hpp>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/dom/DOMImplementationLS.hpp>
#include <xercesc/dom/DOMWriter.hpp>
#include <xercesc/framework/LocalFileFormatTarget.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/sax/InputSource.hpp>
#include <xercesc/sax/HandlerBase.hpp>


#include "FWCore/Utilities/interface/Exception.h"

#include "PhysicsTools/MVATrainer/interface/XMLSimpleStr.h"
#include "PhysicsTools/MVATrainer/interface/XMLUniStr.h"
#include "PhysicsTools/MVATrainer/interface/XMLDocument.h"

XERCES_CPP_NAMESPACE_USE

unsigned int XMLDocument::XercesPlatform::instances = 0;

namespace { // anonymous
	struct DocReleaser {
		inline DocReleaser(DOMDocument *doc) : doc(doc) {}
		inline ~DocReleaser() { doc->release(); }

		DOMDocument *doc;
	};

	template<typename T>
	class XMLInputSourceWrapper :
		public XERCES_CPP_NAMESPACE_QUALIFIER InputSource {
	    public:
		typedef typename T::Stream_t Stream_t;

		XMLInputSourceWrapper(std::auto_ptr<Stream_t> &obj) : obj(obj) {}
		virtual ~XMLInputSourceWrapper() {}

		virtual XERCES_CPP_NAMESPACE_QUALIFIER BinInputStream*
							makeStream() const
		{ return new T(*obj); }

	    private:
		std::auto_ptr<Stream_t>	obj;
	};

	class STLInputStream :
			public XERCES_CPP_NAMESPACE_QUALIFIER BinInputStream {
	    public:
	        typedef std::istream Stream_t;

	        STLInputStream(std::istream &in) : in(in) {}
	        virtual ~STLInputStream() {}

	        virtual unsigned int curPos() const { return pos; }

	        virtual unsigned int readBytes(XMLByte *const buf,
	                                       const unsigned int size);

	    private:
	        std::istream    &in;
	        unsigned int    pos;
	};

	template<int (*close)(FILE*)>
	class stdio_istream : public std::istream {
	    public:
		typedef __gnu_cxx::stdio_filebuf<char>	__filebuf_type;
		typedef stdio_istream<close>		__istream_type;

		stdio_istream(FILE *file) :
			file_(file), filebuf_(file, std::ios_base::in)
		{ this->init(&filebuf_); }

		~stdio_istream()
		{ close(file_); }

		__filebuf_type *rdbuf() const
		{ return const_cast<__filebuf_type*>(&filebuf_); }

	    private:
		FILE		*file_;
		__filebuf_type	filebuf_;
	};

	typedef XMLInputSourceWrapper<STLInputStream> STLInputSource;
} // anonymous namespace

unsigned int STLInputStream::readBytes(XMLByte* const buf,
                                       const unsigned int size)
{
	char *rawBuf = reinterpret_cast<char*>(buf);
	unsigned int bytes = size * sizeof(XMLByte);
	in.read(rawBuf, bytes);
	unsigned int readBytes = in.gcount();

	if (in.bad())
		throw cms::Exception("XMLDocument")
			<< "I/O stream bad in STLInputStream::readBytes()"
			<< std::endl;

	unsigned int read = (unsigned int)(readBytes / sizeof(XMLByte));
	unsigned int rest = (unsigned int)(readBytes % sizeof(XMLByte));
	for(unsigned int i = 1; i <= rest; i++)
		in.putback(rawBuf[readBytes - i]);

	pos += read;
	return read;
}

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

XMLDocument::XMLDocument(const std::string &fileName, bool write) :
	platform(new XercesPlatform()), fileName(fileName),
	write(write), impl(0), doc(0), rootNode(0)
{
	if (write)
		openForWrite(fileName);
	else {
		std::auto_ptr<std::istream> inputStream(
					new std::ifstream(fileName.c_str()));
		if (!inputStream->good())
			throw cms::Exception("XMLDocument")
				<< "XML input file \"" << fileName << "\" "
				   "could not be opened for reading."
				<< std::endl;
		openForRead(inputStream);
	}
}

XMLDocument::XMLDocument(const std::string &fileName,
                         const std::string &command) :
	platform(new XercesPlatform()), fileName(fileName),
	write(false), impl(0), doc(0), rootNode(0)
{
	FILE *file = popen(command.c_str(), "r");
	if (!file)
		throw cms::Exception("XMLDocument")
			<< "Could not execute XML preprocessing "
			   " command \"" << command << "\"."
			<< std::endl;

	std::auto_ptr<std::istream> inputStream(
					new stdio_istream<pclose>(file));
	if (!inputStream->good())
		throw cms::Exception("XMLDocument")
			<< "XML preprocessing command \"" << fileName
			<< "\" stream could not be opened for reading."
			<< std::endl;

	openForRead(inputStream);
}

XMLDocument::~XMLDocument()
{
	if (!write)
		return;

	std::auto_ptr<DocReleaser> docReleaser(new DocReleaser(doc));

	std::auto_ptr<DOMWriter> writer(static_cast<DOMImplementationLS*>(
						impl)->createDOMWriter());
	assert(writer.get());

	writer->setEncoding(XMLUniStr("UTF-8"));
        if (writer->canSetFeature(XMLUni::fgDOMWRTDiscardDefaultContent, true))
		writer->setFeature(XMLUni::fgDOMWRTDiscardDefaultContent, true);
	if (writer->canSetFeature(XMLUni::fgDOMWRTFormatPrettyPrint, true))
		writer->setFeature(XMLUni::fgDOMWRTFormatPrettyPrint, true);

	try {
		std::auto_ptr<XMLFormatTarget> target(
				new LocalFileFormatTarget(fileName.c_str()));

		writer->writeNode(target.get(), *doc);
	} catch(...) {
		std::remove(fileName.c_str());
		throw;
	}
}

void XMLDocument::openForRead(std::auto_ptr<std::istream> &stream)
{
	parser.reset(new XercesDOMParser());
	parser->setValidationScheme(XercesDOMParser::Val_Auto);
	parser->setDoNamespaces(false);
	parser->setDoSchema(false);
	parser->setValidationSchemaFullChecking(false);

	errHandler.reset(new HandlerBase());
	parser->setErrorHandler(errHandler.get());
	parser->setCreateEntityReferenceNodes(false);

	inputSource.reset(new STLInputSource(stream));

	try {
		parser->parse(*inputSource);
		if (parser->getErrorCount())
			throw cms::Exception("XMLDocument")
				<< "XML parser reported errors."
				<< std::endl;
	} catch(const XMLException &e) {
		throw cms::Exception("XMLDocument")
			<< "XML parser reported DOM error no. "
			<< (unsigned long)e.getCode()
			<< ": " << XMLSimpleStr(e.getMessage()) << "."
			<< std::endl;
	} catch(const SAXException &e) {
		throw cms::Exception("XMLDocument")
			<< "XML parser reported: "
			<< XMLSimpleStr(e.getMessage()) << "."
			<< std::endl;
	}

	doc = parser->getDocument();

	DOMNode *node = doc->getFirstChild();
	while(node && node->getNodeType() != DOMNode::ELEMENT_NODE)
		node = node->getNextSibling();

	if (!node)
		throw cms::Exception("XMLDocument")
			<< "XML document didn't contain a valid "
			<< "root node." << std::endl;

	rootNode = static_cast<DOMElement*>(node);
}

void XMLDocument::openForWrite(const std::string &fileName)
{
	impl = DOMImplementationRegistry::getDOMImplementation(
							XMLUniStr("LS"));
	assert(impl);
}

DOMDocument *XMLDocument::createDocument(const std::string &root)
{
	if (doc)
		throw cms::Exception("XMLDocument")
			<< "Document already exists in createDocument."
			<< std::endl;

	doc = impl->createDocument(0, XMLUniStr(root.c_str()), 0);
	rootNode = doc->getDocumentElement();

	return doc;
}

// specialization of read/write method templates for bool

static bool isBool(std::string value)
{
	for(unsigned int i = 0; i < value.size(); i++)
		if (value[i] >= 'A' && value[i] <= 'Z')
			value[i] += 'a' - 'A';

	if (value == "1" || value == "y" || value == "yes" ||
	    value == "true" || value == "ok")
		return true;

	if (value == "0" || value == "n" || value == "no" || value == "false")
		return false;

	throw cms::Exception("XMLDocument")
		<< "Invalid boolean value in XML document" << std::endl;
}

static const char *makeBool(bool value)
{
	return value ? "true" : "false";
}

bool XMLDocument::hasAttribute(XERCES_CPP_NAMESPACE_QUALIFIER DOMElement *elem,
                               const char *name)
{
	XMLUniStr uniName(name);
	return elem->hasAttribute(uniName);
}

template<>
bool XMLDocument::readAttribute(
			XERCES_CPP_NAMESPACE_QUALIFIER DOMElement *elem,
			const char *name)
{
	XMLUniStr uniName(name);
	if (!elem->hasAttribute(uniName))
		throw cms::Exception("MVAComputer")
			<< "Missing attribute " << name << " in tag "
			<< XMLSimpleStr(elem->getNodeName())
			<< "." << std::endl;
	const XMLCh *attribute = elem->getAttribute(uniName);
	return isBool(XMLSimpleStr(attribute));
}

template<>
bool XMLDocument::readAttribute(
			XERCES_CPP_NAMESPACE_QUALIFIER DOMElement *elem,
			const char *name, const bool &defValue)
{
	XMLUniStr uniName(name);
	if (!elem->hasAttribute(uniName))
		return defValue;
	const XMLCh *attribute = elem->getAttribute(uniName);
	return isBool(XMLSimpleStr(attribute));
}

template<>
void XMLDocument::writeAttribute(
			XERCES_CPP_NAMESPACE_QUALIFIER DOMElement *elem,
			const char *name, const bool &value)
{
	elem->setAttribute(XMLUniStr(name), XMLUniStr(makeBool(value)));
}

template<>
bool XMLDocument::readContent(
			XERCES_CPP_NAMESPACE_QUALIFIER DOMNode *node)
{
	const XMLCh *content = node->getTextContent();
	return isBool(XMLSimpleStr(content));
}

template<>
void XMLDocument::writeContent(
			XERCES_CPP_NAMESPACE_QUALIFIER DOMNode *node,
			XERCES_CPP_NAMESPACE_QUALIFIER DOMDocument *doc,
			const bool &value)
{
	node->appendChild(doc->createTextNode(XMLUniStr(makeBool(value))));
}
