#ifndef CaloOnlineTools_HcalOnlineDb_XMLDOMBlock_h
#define CaloOnlineTools_HcalOnlineDb_XMLDOMBlock_h

// -*- C++ -*-
//
// Package:     CaloOnlineTools/HcalOnlineDb
// Class  :     XMLDOMBlock
//
/**\class XMLDOMBlock XMLDOMBlock.h CaloOnlineTools/HcalOnlineDb/interface/XMLDOMBlock.h

 Description: Handle reading bricks from XML file

 Usage:
    <usage>

*/
//
// Original Author:  Gena Kukartsev
//         Created:  Thu Sep 27 01:46:46 CEST 2007
//

#include <xercesc/dom/DOM.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/sax/ErrorHandler.hpp>

#include <memory>
#include <string>

class XMLProcessor;

class XMLDOMBlock {
  friend class XMLProcessor;

public:
  XMLDOMBlock(const std::string& xmlFileName, bool fromScratch = false);

  xercesc::DOMDocument* getDocument(void);
  const xercesc::DOMDocument* getDocumentConst(void) const;
  void write(const std::string& target);
  virtual ~XMLDOMBlock();

  XMLDOMBlock& operator+=(const XMLDOMBlock& other);

protected:
  XMLProcessor* theProcessor = nullptr;
  std::unique_ptr<xercesc::ErrorHandler> errHandler;
  std::unique_ptr<xercesc::XercesDOMParser> parser;
  xercesc::DOMDocument* document = nullptr;
};

#endif
