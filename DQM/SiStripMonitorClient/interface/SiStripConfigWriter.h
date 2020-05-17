#ifndef SiStripConfigWriter_H
#define SiStripConfigWriter_H

/** \class SiStripConfigWriter
 * *
 *  Base class for Parsers used by DQM
 *
 *
 *  \author Suchandra Dutta
 */
#include "Utilities/Xerces/interface/Xerces.h"
#include <xercesc/framework/StdOutFormatTarget.hpp>
#include <xercesc/framework/LocalFileFormatTarget.hpp>

#include <xercesc/util/XMLString.hpp>
#include <xercesc/dom/DOMException.hpp>
#include <xercesc/dom/DOMImplementation.hpp>
#include <xercesc/dom/DOM.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <map>

class SiStripConfigWriter {
public:
  ///Creator
  SiStripConfigWriter();
  ///Destructor
  ~SiStripConfigWriter();
  ///Write XML file
  bool init(const std::string& main);
  void write(const std::string& fname);
  void createElement(const std::string& tag);
  void createElement(const std::string& tag, const std::string& name);
  void createChildElement(const std::string& tag, const std::string& name);
  void createChildElement(const std::string& tag,
                          const std::string& name,
                          const std::string& att_name,
                          const std::string& att_val);
  void createChildElement(const std::string& tag,
                          const std::string& name,
                          const std::string& att_name1,
                          const std::string& att_val1,
                          const std::string& att_name2,
                          const std::string& att_val2);
  void createChildElement(const std::string& tag,
                          const std::string& name,
                          const std::string& att_name1,
                          const std::string& att_val1,
                          const std::string& att_name2,
                          const std::string& att_val2,
                          const std::string& att_name3,
                          const std::string& att_val3);

protected:
private:
  xercesc::DOMElement* theTopElement;
  xercesc::DOMElement* theLastElement;
  xercesc::DOMDocument* theDoc;
  xercesc::DOMLSSerializer* theDomWriter;
  xercesc::DOMLSOutput* theOutput;
};

#endif
