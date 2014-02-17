// -*- C++ -*-
//
// Package:     XMLTools
// Class  :     XMLDOMBlock
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Gena Kukartsev
//         Created:  Thu Sep 27 01:43:42 CEST 2007
// $Id: XMLDOMBlock.cc,v 1.5 2010/08/06 20:24:03 wmtan Exp $
//

// system include files
#include <iostream>
#include <string>
#include<time.h>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/sax/HandlerBase.hpp>
#include <xercesc/dom/DOM.hpp>

//
//_____ following removed as a xalan-c component_____________________
//
//#include <xalanc/XPath/XObject.hpp>

XERCES_CPP_NAMESPACE_USE 
using namespace std;

// user include files
#include "CalibCalorimetry/HcalTPGAlgos/interface/XMLDOMBlock.h"
#include "CalibCalorimetry/HcalTPGAlgos/interface/XMLProcessor.h"



XMLDOMBlock & XMLDOMBlock::operator+=( const XMLDOMBlock & other)
{
  DOMNodeList * _children = other.getDocumentConst()->getDocumentElement()->getChildNodes();
  int _length = _children->getLength();
  //std::cout << "Children nodes:" << _length << std::endl;
  DOMNode * _node;
  for(int i=0;i!=_length;i++){
    _node = _children->item(i);
    DOMNode * i_node = this->getDocument()->importNode(_node,true);
    this->getDocument()->getDocumentElement()->appendChild(i_node);
  }
  return *this;
}


XMLDOMBlock::XMLDOMBlock()
{
  //std::cout << "XMLDOMBlock (or derived): default constructor called - this is meaningless!" << std::endl;
  //std::cout << "XMLDOMBlock (or derived): use yourClass( loaderBaseConfig & ) instead." << std::endl;
  init( "ROOT" );
}

XMLDOMBlock::XMLDOMBlock( std::string _root, int rootElementName )
{
  //std::cout << "XMLDOMBlock (or derived): default constructor called - this is meaningless!" << std::endl;
  //std::cout << "XMLDOMBlock (or derived): use yourClass( loaderBaseConfig & ) instead." << std::endl;
  init( _root );
}


XMLDOMBlock::XMLDOMBlock( InputSource & _source )
{
  //
  //_____ following removed as a xalan-c component_____________________
  //
  /*
  // xalan objects initialization
  theSourceTreeInit = 0;
  theDOMSupport = 0;
  theLiaison = 0;
  theInputSource = 0;
  theDocument = 0;
  thePrefixResolver = 0;
  theEvaluator = 0;
  */

  theProcessor = XMLProcessor::getInstance();

  //theFileName = xmlFileName;

  // initialize the parser
  parser = new XercesDOMParser();
  parser->setValidationScheme(XercesDOMParser::Val_Always);    
  parser->setDoNamespaces(true);    // optional
  
  errHandler = (ErrorHandler*) new HandlerBase();
  parser->setErrorHandler(errHandler);

  // parse the input xml file
  try
    {
      parser->parse( _source );
    }
  catch (const XMLException& toCatch) {
    char* message = XMLString::transcode(toCatch.getMessage());
    std::cout << "Exception message is: \n"
	 << message << "\n";
    XMLString::release(&message);
    //return -1;
  }
  catch (const DOMException& toCatch) {
    char* message = XMLString::transcode(toCatch.msg);
    std::cout << "Exception message is: \n"
	 << message << "\n";
    XMLString::release(&message);
    //return -1;
  }
  catch (...) {
    std::cout << "Unexpected Exception \n" ;
    //return -1;
  }

  //get the XML document
  document = parser -> getDocument();
}


void XMLDOMBlock::parse( InputSource & _source )
{

  theProcessor = XMLProcessor::getInstance();

  //theFileName = xmlFileName;

  // initialize the parser
  parser = new XercesDOMParser();
  parser->setValidationScheme(XercesDOMParser::Val_Always);    
  parser->setDoNamespaces(true);    // optional
  
  errHandler = (ErrorHandler*) new HandlerBase();
  parser->setErrorHandler(errHandler);

  // parse the input xml file
  try
    {
      parser->parse( _source );
    }
  catch (const XMLException& toCatch) {
    char* message = XMLString::transcode(toCatch.getMessage());
    std::cout << "Exception message is: \n"
	 << message << "\n";
    XMLString::release(&message);
    //return -1;
  }
  catch (const DOMException& toCatch) {
    char* message = XMLString::transcode(toCatch.msg);
    std::cout << "Exception message is: \n"
	 << message << "\n";
    XMLString::release(&message);
    //return -1;
  }
  catch (...) {
    std::cout << "Unexpected Exception \n" ;
    //return -1;
  }

  //get the XML document
  document = parser -> getDocument();
}



int XMLDOMBlock::init( std::string _root )
{
  theProcessor = XMLProcessor::getInstance();

  //theFileName = xmlFileName;

  // initialize the parser
  parser = new XercesDOMParser();
  parser->setValidationScheme(XercesDOMParser::Val_Always);    
  parser->setDoNamespaces(true);    // optional
  
  errHandler = (ErrorHandler*) new HandlerBase();
  parser->setErrorHandler(errHandler);

  DOMImplementation* impl =  DOMImplementation::getImplementation();
  
  document = impl->createDocument(
				  0,                      // root element namespace URI.
				  //XMLString::transcode("ROOT"), // root element name
				  XMLProcessor::_toXMLCh(_root), // root element name
				  0);                     // document type object (DTD).

  the_string = 0;

  //
  //_____ following removed as a xalan-c component_____________________
  //
  /*
  // xalan objects initialization
  theSourceTreeInit = 0;
  theDOMSupport = 0;
  theLiaison = 0;
  theInputSource = 0;
  theDocument = 0;
  thePrefixResolver = 0;
  theEvaluator = 0;
  */
  return 0;
}



XMLDOMBlock::XMLDOMBlock( std::string xmlFileName )
{
  //
  //_____ following removed as a xalan-c component_____________________
  //
  /*
  // xalan objects initialization
  theSourceTreeInit = 0;
  theDOMSupport = 0;
  theLiaison = 0;
  theInputSource = 0;
  theDocument = 0;
  thePrefixResolver = 0;
  theEvaluator = 0;
  */

  theProcessor = XMLProcessor::getInstance();

  theFileName = xmlFileName;

  // initialize the parser
  parser = new XercesDOMParser();
  parser->setValidationScheme(XercesDOMParser::Val_Always);    
  parser->setDoNamespaces(true);    // optional
  
  errHandler = (ErrorHandler*) new HandlerBase();
  parser->setErrorHandler(errHandler);

  // parse the input xml file
  try
    {
      parser->parse( theFileName . c_str() );
    }
  catch (const XMLException& toCatch) {
    char* message = XMLString::transcode(toCatch.getMessage());
    std::cout << "Exception message is: \n"
	 << message << "\n";
    XMLString::release(&message);
    //return -1;
  }
  catch (const DOMException& toCatch) {
    char* message = XMLString::transcode(toCatch.msg);
    std::cout << "Exception message is: \n"
	 << message << "\n";
    XMLString::release(&message);
    //return -1;
  }
  catch (...) {
    std::cout << "Unexpected Exception \n" ;
    //return -1;
  }

  //get the XML document
  document = parser -> getDocument();

}

DOMDocument * XMLDOMBlock::getNewDocument( std::string xmlFileName )
{
  delete document;

  theProcessor = XMLProcessor::getInstance();

  theFileName = xmlFileName;

  // initialize the parser
  parser = new XercesDOMParser();
  parser->setValidationScheme(XercesDOMParser::Val_Always);    
  parser->setDoNamespaces(true);    // optional
  
  errHandler = (ErrorHandler*) new HandlerBase();
  parser->setErrorHandler(errHandler);

  // parse the input xml file
  try
    {
      parser->parse( theFileName . c_str() );
    }
  catch (const XMLException& toCatch) {
    char* message = XMLString::transcode(toCatch.getMessage());
    std::cout << "Exception message is: \n"
	 << message << "\n";
    XMLString::release(&message);
    //return -1;
  }
  catch (const DOMException& toCatch) {
    char* message = XMLString::transcode(toCatch.msg);
    std::cout << "Exception message is: \n"
	 << message << "\n";
    XMLString::release(&message);
    //return -1;
  }
  catch (...) {
    std::cout << "Unexpected Exception \n" ;
    //return -1;
  }

  //get the XML document
  document = parser -> getDocument();

  return document;
}

DOMDocument * XMLDOMBlock::getDocument( void )
{
  return document;
}

DOMDocument * XMLDOMBlock::getDocumentConst( void ) const
{
  return document;
}

int XMLDOMBlock::write( std::string target )
{
  theProcessor -> write( this, target );

  return 0;
}

XMLDOMBlock::~XMLDOMBlock()
{
  delete parser;
  delete errHandler;
  //if (the_string) delete the_string;

  //
  //_____ following removed as a xalan-c component_____________________
  //
  /*
  // delete xalan objects
  delete theSourceTreeInit;
  delete theDOMSupport;
  delete theLiaison;
  delete theInputSource;
  //delete theDocument; // noneed to delete - belongs to theLiaison
  delete thePrefixResolver;
  delete theEvaluator;
  */
}

const char * XMLDOMBlock::getTagValue( const std::string & tagName, int _item, DOMDocument * _document )
{
  if (!_document) _document = document;
  const char * _result = XMLString::transcode( _document -> getElementsByTagName( XMLProcessor::_toXMLCh( tagName ) ) -> item( _item ) -> getFirstChild()-> getNodeValue() );
  return _result;
}

const char * XMLDOMBlock::getTagValue( const std::string & tagName, int _item, DOMElement * _document )
{
  if (!_document) return 0;
  const char * _result = XMLString::transcode( _document -> getElementsByTagName( XMLProcessor::_toXMLCh( tagName ) ) -> item( _item ) -> getFirstChild()-> getNodeValue() );
  return _result;
}

DOMNode * XMLDOMBlock::setTagValue( const std::string & tagName, const std::string & tagValue, int _item, DOMDocument * _document )
{
  if (!_document) _document = document;
  DOMNode * the_tag = _document -> getElementsByTagName( XMLProcessor::_toXMLCh( tagName ) ) -> item( _item );
  the_tag -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( tagValue ) );
  return the_tag;
}

DOMNode * XMLDOMBlock::setTagValue(DOMElement * _elem, const std::string & tagName, const std::string & tagValue, int _item )
{
  if (!_elem) return 0;
  DOMNode * the_tag = _elem -> getElementsByTagName( XMLProcessor::_toXMLCh( tagName ) ) -> item( _item );
  if (the_tag){
    the_tag -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( tagValue ) );
  }
  return the_tag;
}

DOMNode * XMLDOMBlock::setTagValue( const std::string & tagName, const int & tagValue, int _item, DOMDocument * _document )
{
  if (!_document) _document = document;
  DOMNode * the_tag = _document -> getElementsByTagName( XMLProcessor::_toXMLCh( tagName ) ) -> item( _item );
  the_tag -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( tagValue ) );
  return the_tag;
}

DOMNode * XMLDOMBlock::setTagValue( DOMElement * _elem, const std::string & tagName, const int & tagValue, int _item )
{
  if (!_elem) return 0;
  DOMNode * the_tag = _elem -> getElementsByTagName( XMLProcessor::_toXMLCh( tagName ) ) -> item( _item );
  if(the_tag){
    the_tag -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( tagValue ) );
  }
  return the_tag;
}

const char * XMLDOMBlock::getTagAttribute( const std::string & tagName, const std::string & attrName, int _item )
{
  DOMElement * _tag = (DOMElement *)(document -> getElementsByTagName( XMLProcessor::_toXMLCh( tagName ) ) -> item( _item ));
  const char * _result = XMLString::transcode( _tag -> getAttribute( XMLProcessor::_toXMLCh( attrName ) ) );

  return _result;
}

DOMNode * XMLDOMBlock::setTagAttribute( const std::string & tagName, const std::string & attrName, const std::string & attrValue, int _item )
{
  DOMNode * the_tag = document -> getElementsByTagName( XMLProcessor::_toXMLCh( tagName ) ) -> item( _item );
  DOMElement * _tag = (DOMElement *)the_tag;
  _tag -> setAttribute( XMLProcessor::_toXMLCh( attrName ), XMLProcessor::_toXMLCh( attrValue ) );
  return the_tag;
}

DOMNode * XMLDOMBlock::setTagAttribute( DOMElement * _elem, const std::string & tagName, const std::string & attrName, const std::string & attrValue, int _item )
{
  if (!_elem) return 0;
  DOMNode * the_tag = _elem ->  getElementsByTagName( XMLProcessor::_toXMLCh( tagName ) ) -> item( _item );
  if (the_tag){
    DOMElement * _tag = (DOMElement *)the_tag;
    _tag -> setAttribute( XMLProcessor::_toXMLCh( attrName ), XMLProcessor::_toXMLCh( attrValue ) );
  }
  return the_tag;
}

DOMNode * XMLDOMBlock::setTagAttribute( const std::string & tagName, const std::string & attrName, const int & attrValue, int _item )
{
  DOMNode * the_tag = document -> getElementsByTagName( XMLProcessor::_toXMLCh( tagName ) ) -> item( _item );
  DOMElement * _tag = (DOMElement *)the_tag;
  _tag -> setAttribute( XMLProcessor::_toXMLCh( attrName ), XMLProcessor::_toXMLCh( attrValue ) );
  return the_tag;
}

DOMNode * XMLDOMBlock::setTagAttribute( DOMElement * _elem, const std::string & tagName, const std::string & attrName, const int & attrValue, int _item )
{
  if (!_elem) return 0;
  DOMNode * the_tag = _elem -> getElementsByTagName( XMLProcessor::_toXMLCh( tagName ) ) -> item( _item );
  if (the_tag){
    DOMElement * _tag = (DOMElement *)the_tag;
    _tag -> setAttribute( XMLProcessor::_toXMLCh( attrName ), XMLProcessor::_toXMLCh( attrValue ) );
  }
  return the_tag;
}

string XMLDOMBlock::getTimestamp( time_t _time )
{
  char timebuf[50];
  //strftime( timebuf, 50, "%c", gmtime( &_time ) );
  strftime( timebuf, 50, "%Y-%m-%d %H:%M:%S.0", gmtime( &_time ) );
  std::string creationstamp = timebuf;

  return creationstamp;
}




std::string & XMLDOMBlock::getString( void )
{
  return getString( this->getDocument() );
}




std::string & XMLDOMBlock::getString( DOMNode * _node )
{
  if (the_string) delete the_string;
  std::string _target = "string";
  the_string = new std::string( XMLString::transcode( theProcessor->serializeDOM(_node,_target) ) );
  return (*the_string);
}


//
//_____ following removed as a xalan-c component_____________________
//
/*
int XMLDOMBlock::read_xml_file_xalan( std::string filename ){
  theSourceTreeInit = new XalanSourceTreeInit();
  theDOMSupport = new XalanSourceTreeDOMSupport();
  theLiaison = new XalanSourceTreeParserLiaison(*theDOMSupport);
  const XalanDOMString theFileName(filename.c_str());
  theInputSource = new LocalFileInputSource(theFileName.c_str());
  theDocument = theLiaison->parseXMLStream(*theInputSource);
  assert(theDocument != 0);
  thePrefixResolver = new XalanDocumentPrefixResolver(theDocument);
  theEvaluator = new XPathEvaluator;
  return 0;
}
*/

//
//_____ following removed as a xalan-c component_____________________
//
/*
const XObjectPtr XMLDOMBlock::eval_xpath( std::string context, std::string expression ){
  XalanNode* const theContextNode = theEvaluator->selectSingleNode(
								   *theDOMSupport,
								  theDocument,
								  XalanDOMString(context.c_str()).c_str(),
								  *thePrefixResolver);
  if (theContextNode == 0){
    //std::cerr << "Warning -- No nodes matched the location path " << context << std::endl;
    XObjectPtr _null;
    return _null;
  }

  
  const XObjectPtr theResult(
			     theEvaluator->evaluate(
						    *theDOMSupport,
						    theContextNode,
						    XalanDOMString(expression.c_str()).c_str(),
						    *thePrefixResolver));
  return theResult;
}
*/



DOMElement * XMLDOMBlock::add_element(DOMElement * parent, XMLCh * tagname, XMLCh * value){
  DOMElement * _elem = document -> createElement( tagname );
  parent -> appendChild(_elem);
  DOMText * _value = document -> createTextNode(value);
  _elem->appendChild(_value);
  return _elem;
}


