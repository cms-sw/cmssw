#ifndef PhysicsTools_MVATrainer_XMLDocument_h
#define PhysicsTools_MVATrainer_XMLDocument_h

#include <string>
#include <memory>
#include <iosfwd>

#include <xercesc/dom/DOM.hpp>
#include <xercesc/dom/DOMImplementation.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/sax/HandlerBase.hpp>
#include <xercesc/sax/InputSource.hpp>
#include <xercesc/dom/DOMDocument.hpp>

class XMLDocument {
    public:
	XMLDocument(const std::string &fileName, bool write = false);
	XMLDocument(const std::string &fileName, const std::string &command);
	~XMLDocument();

	inline XERCES_CPP_NAMESPACE_QUALIFIER DOMDocument *
	getDocument() const { return doc; }

	inline XERCES_CPP_NAMESPACE_QUALIFIER DOMElement *
	getRootNode() const { return rootNode; }

	XERCES_CPP_NAMESPACE_QUALIFIER DOMDocument *
	createDocument(const std::string &root);

	static bool hasAttribute(XERCES_CPP_NAMESPACE_QUALIFIER DOMElement *elem,
	                         const char *name);
	template<typename T>
	static T readAttribute(XERCES_CPP_NAMESPACE_QUALIFIER DOMElement *elem,
	                       const char *name);
	template<typename T>
	static T readAttribute(XERCES_CPP_NAMESPACE_QUALIFIER DOMElement *elem,
	                       const char *name, const T &defValue);
	template<typename T>
	static void writeAttribute(XERCES_CPP_NAMESPACE_QUALIFIER DOMElement *elem,
	                           const char *name, const T &value);

	template<typename T>
	static T readContent(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode *node);
	template<typename T>
	static void writeContent(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode *node,
			XERCES_CPP_NAMESPACE_QUALIFIER DOMDocument *doc,
			const T &value);

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

	void openForRead(std::auto_ptr<std::istream> &inputStream);
	void openForWrite(const std::string &fileName);

	std::auto_ptr<XercesPlatform>					platform;
	std::auto_ptr<XERCES_CPP_NAMESPACE_QUALIFIER InputSource>	inputSource;

	std::string							fileName;
	bool								write;

	std::auto_ptr<XERCES_CPP_NAMESPACE_QUALIFIER XercesDOMParser>	parser;
	std::auto_ptr<XERCES_CPP_NAMESPACE_QUALIFIER HandlerBase>	errHandler;
	XERCES_CPP_NAMESPACE_QUALIFIER DOMImplementation		*impl;

	XERCES_CPP_NAMESPACE_QUALIFIER DOMDocument			*doc;
	XERCES_CPP_NAMESPACE_QUALIFIER DOMElement			*rootNode;
};

#include "PhysicsTools/MVATrainer/interface/XMLDocument.icc"

#endif // PhysicsTools_MVATrainer_XMLDocument_h
