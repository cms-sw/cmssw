#ifndef HCALConfigDBTools_XMLTools_XMLDOMBlock_h
#define HCALConfigDBTools_XMLTools_XMLDOMBlock_h
// -*- C++ -*-
//
// Package:     XMLTools
// Class  :     XMLDOMBlock
// 
/**\class XMLDOMBlock XMLDOMBlock.h CaloOnlineTools/HcalOnlineDb/interface/XMLDOMBlock.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Gena Kukartsev
//         Created:  Thu Sep 27 01:46:46 CEST 2007
// $Id: XMLDOMBlock.h,v 1.3 2007/12/06 02:26:09 kukartse Exp $
//

// system include files
#include <string>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/sax/HandlerBase.hpp>
#include <xercesc/dom/DOM.hpp>

XERCES_CPP_NAMESPACE_USE 
using namespace std;

// user include files

// forward declarations
class XMLProcessor;

class XMLDOMBlock
{

  friend class XMLProcessor;
  
 public:

  XMLDOMBlock();
  XMLDOMBlock( string xmlFileName );
  XMLDOMBlock( InputSource & _source );

  DOMDocument * getDocument( void );
  DOMDocument * getNewDocument( string xmlFileName );
  int write( string target = "stdout" );
  virtual ~XMLDOMBlock();
  
  // ---------- const member functions ---------------------
  
  // ---------- static member functions --------------------
  
  // ---------- member functions ---------------------------
  const char * getTagValue( const string & tagName, int _item = 0, DOMDocument * _document = NULL );
  const char * getTagAttribute( const string & tagName, const string & attrName, int _item = 0 );

  int setTagValue( const string & tagName, const string & tagValue, int _item = 0, DOMDocument * _document = NULL );
  int setTagValue( const string & tagName, const int & tagValue, int _item = 0, DOMDocument * _document = NULL );
  int setTagAttribute( const string & tagName, const string & attrName, const string & attrValue, int _item = 0 );
  string getTimestamp( time_t _time );  
 protected:
  //XMLDOMBlock(const XMLDOMBlock&); // stop default
  
  //const XMLDOMBlock& operator=(const XMLDOMBlock&); // stop default
  
  // ---------- member data --------------------------------
  XMLProcessor * theProcessor;
  XercesDOMParser * parser;
  ErrorHandler * errHandler;
  DOMDocument * document;
  string theFileName;
};


#endif
