#ifndef HCALConfigDBTools_XMLTools_XMLDOMBlock_h
#define HCALConfigDBTools_XMLTools_XMLDOMBlock_h
// -*- C++ -*-
//
// Package:     XMLTools
// Class  :     XMLDOMBlock
//
/**\class XMLDOMBlock XMLDOMBlock.h CalibCalorimetry/HcalTPGAlgos/interface/XMLDOMBlock.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Gena Kukartsev
//         Created:  Thu Sep 27 01:46:46 CEST 2007
//

#include <string>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/sax/HandlerBase.hpp>
#include <xercesc/dom/DOM.hpp>

//
//_____ following removed as a xalan-c component_____________________
//
//#include <xalanc/DOMSupport/XalanDocumentPrefixResolver.hpp>
////#include <xalanc/XPath/XObject.hpp>
//#include <xalanc/XalanSourceTree/XalanSourceTreeDOMSupport.hpp>
//#include <xalanc/XalanSourceTree/XalanSourceTreeInit.hpp>
//#include <xalanc/XalanSourceTree/XalanSourceTreeParserLiaison.hpp>
//#include <xercesc/framework/LocalFileInputSource.hpp>
//#include <xalanc/XPath/XPathEvaluator.hpp>

class XMLProcessor;

class XMLDOMBlock {
  friend class XMLProcessor;

public:
  XMLDOMBlock();
  XMLDOMBlock(std::string xmlFileName);  // create XML from template file
  XMLDOMBlock(XERCES_CPP_NAMESPACE::InputSource& _source);
  XMLDOMBlock(std::string _root, int rootElementName);  // create XML from scratch, second parameter is a dummy

  XERCES_CPP_NAMESPACE::DOMDocument* getDocument(void);
  XERCES_CPP_NAMESPACE::DOMDocument* getDocumentConst(void) const;
  XERCES_CPP_NAMESPACE::DOMDocument* getNewDocument(std::string xmlFileName);
  std::string& getString(void);
  std::string& getString(XERCES_CPP_NAMESPACE::DOMNode* _node);
  int write(std::string target = "stdout");
  virtual ~XMLDOMBlock();

  const char* getTagValue(const std::string& tagName,
                          int _item = 0,
                          XERCES_CPP_NAMESPACE::DOMDocument* _document = nullptr);
  const char* getTagValue(const std::string& tagName, int _item, XERCES_CPP_NAMESPACE::DOMElement* _document);
  const char* getTagAttribute(const std::string& tagName, const std::string& attrName, int _item = 0);

  XERCES_CPP_NAMESPACE::DOMElement* add_element(XERCES_CPP_NAMESPACE::DOMElement* parent, XMLCh* tagname, XMLCh* value);

  XERCES_CPP_NAMESPACE::DOMNode* setTagValue(const std::string& tagName,
                                             const std::string& tagValue,
                                             int _item = 0,
                                             XERCES_CPP_NAMESPACE::DOMDocument* _document = nullptr);
  XERCES_CPP_NAMESPACE::DOMNode* setTagValue(XERCES_CPP_NAMESPACE::DOMElement* _elem,
                                             const std::string& tagName,
                                             const std::string& tagValue,
                                             int _item = 0);
  XERCES_CPP_NAMESPACE::DOMNode* setTagValue(const std::string& tagName,
                                             const int& tagValue,
                                             int _item = 0,
                                             XERCES_CPP_NAMESPACE::DOMDocument* _document = nullptr);
  XERCES_CPP_NAMESPACE::DOMNode* setTagValue(XERCES_CPP_NAMESPACE::DOMElement* _elem,
                                             const std::string& tagName,
                                             const int& tagValue,
                                             int _item = 0);
  XERCES_CPP_NAMESPACE::DOMNode* setTagAttribute(const std::string& tagName,
                                                 const std::string& attrName,
                                                 const std::string& attrValue,
                                                 int _item = 0);
  XERCES_CPP_NAMESPACE::DOMNode* setTagAttribute(XERCES_CPP_NAMESPACE::DOMElement* _elem,
                                                 const std::string& tagName,
                                                 const std::string& attrName,
                                                 const std::string& attrValue,
                                                 int _item = 0);
  XERCES_CPP_NAMESPACE::DOMNode* setTagAttribute(const std::string& tagName,
                                                 const std::string& attrName,
                                                 const int& attrValue,
                                                 int _item = 0);
  XERCES_CPP_NAMESPACE::DOMNode* setTagAttribute(XERCES_CPP_NAMESPACE::DOMElement* _elem,
                                                 const std::string& tagName,
                                                 const std::string& attrName,
                                                 const int& attrValue,
                                                 int _item = 0);
  std::string getTimestamp(time_t _time);

  void parse(XERCES_CPP_NAMESPACE::InputSource& _source);

  XMLDOMBlock& operator+=(const XMLDOMBlock& other);

  //
  //_____ following removed as a xalan-c component_____________________
  //
  //===> Xalan-c (XPath) stuff
  //int read_xml_file_xalan( std::string filename );
  //const XObjectPtr eval_xpath( std::string context, std::string expression );

protected:
  int init(std::string _root);

  XMLProcessor* theProcessor;
  XERCES_CPP_NAMESPACE::XercesDOMParser* parser;
  XERCES_CPP_NAMESPACE::ErrorHandler* errHandler;
  XERCES_CPP_NAMESPACE::DOMDocument* document;
  std::string theFileName;
  std::string* the_string;

  //
  //_____ following removed as a xalan-c component_____________________
  //
  // xalan objects for XPath
  //XalanSourceTreeInit * theSourceTreeInit;
  //XalanSourceTreeDOMSupport * theDOMSupport;
  //XalanSourceTreeParserLiaison * theLiaison;
  //const LocalFileInputSource * theInputSource;
  //XalanDocument * theDocument;
  //XalanDocumentPrefixResolver * thePrefixResolver;
  //XPathEvaluator * theEvaluator;
};

#endif
