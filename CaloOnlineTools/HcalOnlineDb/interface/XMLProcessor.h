#ifndef CaloOnlineTools_HcalOnlineDb_XMLProcessor_h
#define CaloOnlineTools_HcalOnlineDb_XMLProcessor_h

// -*- C++ -*-
//
// Package:     CaloOnlineTools/HcalOnlineDb
// Class  :     XMLProcessor
//
/**\class XMLProcessor XMLProcessor.h CaloOnlineTools/HcalOnlineDb/interface/XMLProcessor.h

 Description: Helper class for serializing and writing HCAL LUT XML

 Usage:
    <usage>

*/
//
// Original Author:  Gena Kukartsev
//         Created:  Sun Sep 23 16:57:06 CEST 2007
//

#include "CaloOnlineTools/HcalOnlineDb/interface/XMLDOMBlock.h"

#include <xercesc/dom/DOM.hpp>
#include <xercesc/util/XMLException.hpp>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/util/XMLUni.hpp>

#include <cstdio>
#include <ctime>
#include <iostream>
#include <memory>
#include <string>

class XMLProcessor {
public:
  // this class is a singleton
  static XMLProcessor* getInstance() {
    if (!instance)
      instance = new XMLProcessor();
    return instance;
  }

  XMLProcessor(const XMLProcessor&) = delete;  // stop default

  struct XMLChDeleter {
    void operator()(XMLCh* ptr) const noexcept { xercesc::XMLString::release(&ptr); }
  };

  void write(XMLDOMBlock* doc, const std::string& target);

  void serializeDOM(xercesc::DOMNode* node, const std::string& target);

  static std::unique_ptr<XMLCh, XMLChDeleter> _toXMLCh(const std::string& temp);
  static std::unique_ptr<XMLCh, XMLChDeleter> _toXMLCh(const int temp);

  virtual ~XMLProcessor();

  void init(void);
  void terminate(void);

private:
  XMLProcessor();

  static XMLProcessor* instance;
};

#endif
