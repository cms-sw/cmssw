#ifndef PhysicsTools_MVATrainer_XMLDocument_h
#define PhysicsTools_MVATrainer_XMLDocument_h

#include <iomanip>
#include <sstream>
#include <string>
#include <memory>

#include <xercesc/dom/DOM.hpp>
#include <xercesc/dom/DOMImplementation.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/sax/HandlerBase.hpp>
#include <xercesc/dom/DOMDocument.hpp>

#include "FWCore/Utilities/interface/Exception.h"

#include "PhysicsTools/MVATrainer/interface/XMLSimpleStr.h"
#include "PhysicsTools/MVATrainer/interface/XMLUniStr.h"

class XMLDocument {
    public:
	XMLDocument(const std::string &fileName, bool write = false);
	~XMLDocument();

	inline XERCES_CPP_NAMESPACE_QUALIFIER DOMDocument *
	getDocument() const { return doc; }

	inline XERCES_CPP_NAMESPACE_QUALIFIER DOMElement *
	getRootNode() const { return rootNode; }

	XERCES_CPP_NAMESPACE_QUALIFIER DOMDocument *
	createDocument(const std::string &root);

	template<typename T>
	static T readAttribute(XERCES_CPP_NAMESPACE_QUALIFIER DOMElement *elem,
	                       const char *name)
	{
		XMLUniStr uniName(name);
		if (!elem->hasAttribute(uniName))
			throw cms::Exception("MVAComputer")
				<< "Missing attribute " << name << " in tag "
				<< XMLSimpleStr(elem->getNodeName())
				<< "." << std::endl;
		const XMLCh *attribute = elem->getAttribute(uniName);
		T value = T();
		std::istringstream buffer((const char*)XMLSimpleStr(attribute));
		buffer >> value;
		return value;
	}

	template<typename T>
	static T readAttribute(XERCES_CPP_NAMESPACE_QUALIFIER DOMElement *elem,
	                       const char *name, const T &defValue)
	{
		XMLUniStr uniName(name);
		if (!elem->hasAttribute(uniName))
			return defValue;
		const XMLCh *attribute = elem->getAttribute(uniName);
		std::istringstream buffer((const char*)XMLSimpleStr(attribute));
		T value = defValue;
		buffer >> value;
		return value;
	}

	template<typename T>
	static void writeAttribute(XERCES_CPP_NAMESPACE_QUALIFIER DOMElement *elem,
	                           const char *name, const T &value)
	{
		std::ostringstream os;
		os << std::setprecision(16) << value;
		elem->setAttribute(XMLUniStr(name),
		                   XMLUniStr(os.str().c_str()));
	}

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

	void openForRead(const std::string &fileName);
	void openForWrite(const std::string &fileName);

	std::auto_ptr<XercesPlatform>					platform;

	std::string							fileName;
	bool								write;

	std::auto_ptr<XERCES_CPP_NAMESPACE_QUALIFIER XercesDOMParser>	parser;
	std::auto_ptr<XERCES_CPP_NAMESPACE_QUALIFIER HandlerBase>	errHandler;
	XERCES_CPP_NAMESPACE_QUALIFIER DOMImplementation		*impl;

	XERCES_CPP_NAMESPACE_QUALIFIER DOMDocument			*doc;
	XERCES_CPP_NAMESPACE_QUALIFIER DOMElement			*rootNode;
};

#endif // PhysicsTools_MVATrainer_XMLDocument_h
