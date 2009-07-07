#include <assert.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <memory>
#include <string>

#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/util/XMLUni.hpp>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/dom/DOMImplementationLS.hpp>
#include <xercesc/dom/DOMWriter.hpp>
#include <xercesc/framework/LocalFileFormatTarget.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
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
} // anonymous namespace

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
	else
		openForRead(fileName);
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

void XMLDocument::openForRead(const std::string &fileName)
{
	parser = std::auto_ptr<XercesDOMParser>(new XercesDOMParser());
	parser->setValidationScheme(XercesDOMParser::Val_Auto);
	parser->setDoNamespaces(false);
	parser->setDoSchema(false);
	parser->setValidationSchemaFullChecking(false);

	errHandler = std::auto_ptr<HandlerBase>(new HandlerBase());
	parser->setErrorHandler(errHandler.operator -> ());
	parser->setCreateEntityReferenceNodes(false);

	try {
		parser->parse(fileName.c_str());
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
