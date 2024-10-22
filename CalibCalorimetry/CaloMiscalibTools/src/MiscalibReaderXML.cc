#include "CalibCalorimetry/CaloMiscalibTools/interface/MiscalibReaderFromXML.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/MiscalibReaderFromXMLDomUtils.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/CaloMiscalibMap.h"
#include "Utilities/Xerces/interface/Xerces.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace xercesc;

int MiscalibReaderFromXML::s_numberOfInstances = 0;  //to check that there is only 1 instance

inline std::string _toString(const XMLCh* toTranscode) {
  std::string tmp(XMLString::transcode(toTranscode));
  return tmp;
}

inline XMLCh* _toDOMS(std::string temp) {
  XMLCh* buff = XMLString::transcode(temp.c_str());
  return buff;
}

MiscalibReaderFromXML::MiscalibReaderFromXML(CaloMiscalibMap& caloMap) : caloMap_(caloMap) {
  try {
    //std::cout << "Xerces-c initialization Number "
    //<< s_numberOfInstances<<std::endl;
    if (s_numberOfInstances == 0)
      cms::concurrency::xercesInitialize();
  } catch (const XMLException& e) {
    std::cout << "Xerces-c error in initialization \n"
              << "Exception message is:  \n"
              << _toString(e.getMessage()) << std::endl;
    // throw an exception here
  }

  ++s_numberOfInstances;
}
//////////////////////////////////////////////////////////////////////////////////

int MiscalibReaderFromXML::getIntAttribute(DOMNamedNodeMap* attribute, const std::string& attribute_name) {
  bool well_formed_string;
  int retval = MiscalibReaderFromXMLDomUtils::getIntAttribute(attribute, attribute_name, well_formed_string);
  if (!well_formed_string)
    std::cout << "MiscalibReaderFromXML::getIntAttribute PROBLEMS ...!!!" << std::endl;

  return retval;
}

//////////////////////////////////////////////////////////////////////////////////

double MiscalibReaderFromXML::getScalingFactor(XERCES_CPP_NAMESPACE::DOMNamedNodeMap* attribute) {
  return MiscalibReaderFromXML::getFloatAttribute(attribute, "scale_factor");
}

//////////////////////////////////////////////////////////////////////////////////

double MiscalibReaderFromXML::getFloatAttribute(DOMNamedNodeMap* attribute, const std::string& attribute_name) {
  bool well_formed_string;
  double retval = MiscalibReaderFromXMLDomUtils::getFloatAttribute(attribute, attribute_name, well_formed_string);
  if (!well_formed_string)
    std::cout << "MiscalibReaderFromXML::getFloatAttribute PROBLEMS ...!!!" << std::endl;

  return retval;
}

//////////////////////////////////////////////////////////////////////////////////

bool MiscalibReaderFromXML::parseXMLMiscalibFile(std::string configFile) {
  XercesDOMParser* parser = new XercesDOMParser;
  parser->setValidationScheme(XercesDOMParser::Val_Auto);
  parser->setDoNamespaces(false);
  parser->parse(configFile.c_str());
  DOMDocument* doc = parser->getDocument();
  assert(doc);

  unsigned int linkTagsNum = doc->getElementsByTagName(_toDOMS("Cell"))->getLength();
  // The following should be on LogInfo
  //std::cout << "Read number of Cells = " << linkTagsNum << std::endl;

  if (linkTagsNum == 0)
    std::cout << "Number of Cells in file is 0 - probably bad file format" << std::endl;

  for (unsigned int i = 0; i < linkTagsNum; i++) {
    DOMNode* linkNode = doc->getElementsByTagName(_toDOMS("Cell"))->item(i);
    ///Get ME name
    if (!linkNode) {
      std::cout << "Node LINK does not exist, i=" << i << std::endl;
      return true;
    }
    DOMElement* linkElement = static_cast<DOMElement*>(linkNode);
    if (!linkElement) {
      std::cout << "Element LINK does not exist, i=" << i << std::endl;
      return true;
    }

    DOMNamedNodeMap* attributes = linkNode->getAttributes();
    double scalingfactor = getScalingFactor(attributes);

    DetId cell = parseCellEntry(attributes);

    if (cell != DetId(0)) {
      caloMap_.addCell(cell, scalingfactor);
    } else {
      //		std::cout << "Null received" << std::endl;
    }
  }

  // The following should be on LogInfo
  // std::cout << "Number of good Cells = " << count << std::endl;
  return false;
}
