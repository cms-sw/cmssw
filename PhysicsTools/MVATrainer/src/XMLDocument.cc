#include <assert.h>
#include <iostream>
#include <cstdio>
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
		} catch(...) {
			throw cms::Exception("MVATrainer")
				<< "XMLPlatformUtils::Initialize failed."
				<< std::endl;
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
			throw cms::Exception("MVATrainer")
				<< "XML parser reported errors."
				<< std::endl;
	} catch(const XMLException &e) {
		throw cms::Exception("MVATrainer")
			<< "XML parser reported DOM error no. "
			<< (unsigned long)e.getCode()
			<< "." << std::endl;
	} catch(...) {
		throw cms::Exception("MVATrainer")
			<< "XML parser reported an unknown error."
			<< std::endl;
	}

	doc = parser->getDocument();

	DOMNode *node = doc->getFirstChild();
	while(node && node->getNodeType() != DOMNode::ELEMENT_NODE)
		node = node->getNextSibling();

	if (!node)
		throw cms::Exception("MVATrainer")
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
