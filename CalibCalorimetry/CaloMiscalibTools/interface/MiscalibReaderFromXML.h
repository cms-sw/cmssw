#ifndef MiscalibReaderFromXML_H
#define MiscalibReaderFromXML_H

/** \class MiscalibReaderFromXML
 * *
 *  Parses the xml file to get miscalibration constants
 * 
 *  $Date: 2010/08/07 14:55:53 $
 *  $Revision: 1.2 $
 *  \author Lorenzo Agostino
  */

#include <xercesc/dom/DOMNode.hpp>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/dom/DOMCharacterData.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/framework/LocalFileFormatTarget.hpp>
#include <xercesc/util/XMLUni.hpp>
#include <xercesc/util/XMLURL.hpp>
#include "CalibCalorimetry/CaloMiscalibTools/interface/CaloMiscalibMap.h"
          

#include<iostream>
#include<string>
#include<vector>
#include<map>


class MiscalibReaderFromXML{

public:
MiscalibReaderFromXML(CaloMiscalibMap &);
virtual ~MiscalibReaderFromXML(){}

bool parseXMLMiscalibFile(std::string configFile);

virtual DetId parseCellEntry(XERCES_CPP_NAMESPACE::DOMNamedNodeMap *attribute)=0;
int    getIntAttribute(XERCES_CPP_NAMESPACE::DOMNamedNodeMap *attribute, const std::string &attribute_name);
double getScalingFactor(XERCES_CPP_NAMESPACE::DOMNamedNodeMap *attribute);
double getFloatAttribute(XERCES_CPP_NAMESPACE::DOMNamedNodeMap *attribute, const std::string &attribute_name);

private:
static int s_numberOfInstances;
CaloMiscalibMap & caloMap_;
};

#endif
