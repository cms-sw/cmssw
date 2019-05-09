#ifndef SiPixelConfigWriter_H
#define SiPixelConfigWriter_H

/** \class SiPixelConfigWriter
 * *
 *  Base class for Parsers used by DQM
 *
 *
 *  \author Petra Merkel
 */
#include "FWCore/Concurrency/interface/Xerces.h"
#include <xercesc/framework/LocalFileFormatTarget.hpp>
#include <xercesc/framework/StdOutFormatTarget.hpp>

#include <xercesc/dom/DOM.hpp>
#include <xercesc/dom/DOMException.hpp>
#include <xercesc/dom/DOMImplementation.hpp>
#include <xercesc/util/XMLString.hpp>

#include <iostream>
#include <map>
#include <string>
#include <vector>

class SiPixelConfigWriter {
public:
  /// Creator
  SiPixelConfigWriter();
  /// Destructor
  ~SiPixelConfigWriter();
  /// Write XML file
  bool init();
  void write(std::string &fname);
  void createLayout(std::string &name);
  void createRow();
  void createColumn(std::string &element, std::string &name);

protected:
private:
  xercesc::DOMElement *theTopElement;
  xercesc::DOMElement *theLastLayout;
  xercesc::DOMElement *theLastRow;
  xercesc::DOMDocument *theDoc;
  xercesc::DOMLSSerializer *theDomWriter;
  xercesc::DOMLSOutput *theOutput;
};

#endif
