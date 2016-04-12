#ifndef OMTF_XMLConfigReader_H
#define OMTF_XMLConfigReader_H

#include <string>
#include <vector>
#include <ostream>

#include "xercesc/util/XercesDefs.hpp"
#include "xercesc/dom/DOM.hpp"

#include "CondFormats/L1TObjects/interface/LUT.h"
#include "L1Trigger/L1TMuonOverlap/interface/OMTFConfiguration.h"

class GoldenPattern;
class OMTFConfiguration;
class L1TMuonOverlapParams;


namespace XERCES_CPP_NAMESPACE{

class DOMElement;
class XercesDOMParser;

}

class XMLConfigReader{

 public:

  XMLConfigReader(OMTFConfiguration * omtf_config);

  void readConfig(const std::string fName);

  void setConfigFile(const std::string & fName) {configFile = fName;}

  void setPatternsFile(const std::string & fName) {patternsFile = fName;}

  void setEventsFile(const std::string & fName) {eventsFile = fName;}

  std::vector<GoldenPattern*> readPatterns();

  void readLUT(l1t::LUT *lut, const std::string & type);

  void readConfig(OMTFConfiguration *aConfig);

  void readConfig(L1TMuonOverlapParams *aConfig);

  std::vector<std::vector<int> > readEvent(unsigned int iEvent=0,
					   unsigned int iProcessor=0,
					   bool readEta = false);

 private:

  std::string configFile; //XML file with general configuration
  std::string patternsFile; //XML file with GoldenPatterns
  std::string eventsFile;   //XML file with events

  GoldenPattern * buildGP(xercesc::DOMElement* aGPElement,
			  unsigned int index=0);
  
  xercesc::XercesDOMParser *parser;
  xercesc::DOMDocument* doc;

  ///Cache with GPs read.
  std::vector<GoldenPattern*> aGPs;

  OMTFConfiguration * m_omtf_config;
};


//////////////////////////////////
//////////////////////////////////
#endif
