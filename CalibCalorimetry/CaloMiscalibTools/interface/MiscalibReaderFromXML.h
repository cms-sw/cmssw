#ifndef MiscalibReaderFromXML_H
#define MiscalibReaderFromXML_H

/** \class MiscalibReaderFromXML
 * *
 *  Parses the xml file to get miscalibration constants
 *
 *  \author Lorenzo Agostino
 */

#include "CalibCalorimetry/CaloMiscalibTools/interface/CaloMiscalibMap.h"
#include "FWCore/Concurrency/interface/Xerces.h"
#include <xercesc/dom/DOM.hpp>
#include <xercesc/dom/DOMCharacterData.hpp>
#include <xercesc/dom/DOMNode.hpp>
#include <xercesc/framework/LocalFileFormatTarget.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/util/XMLURL.hpp>
#include <xercesc/util/XMLUni.hpp>

#include <iostream>
#include <map>
#include <string>
#include <vector>

class MiscalibReaderFromXML {

public:
  MiscalibReaderFromXML(CaloMiscalibMap &);
  virtual ~MiscalibReaderFromXML() {}

  bool parseXMLMiscalibFile(std::string configFile);

  virtual DetId
  parseCellEntry(XERCES_CPP_NAMESPACE::DOMNamedNodeMap *attribute) = 0;
  int getIntAttribute(XERCES_CPP_NAMESPACE::DOMNamedNodeMap *attribute,
                      const std::string &attribute_name);
  double getScalingFactor(XERCES_CPP_NAMESPACE::DOMNamedNodeMap *attribute);
  double getFloatAttribute(XERCES_CPP_NAMESPACE::DOMNamedNodeMap *attribute,
                           const std::string &attribute_name);

private:
  static int s_numberOfInstances;
  CaloMiscalibMap &caloMap_;
};

#endif
