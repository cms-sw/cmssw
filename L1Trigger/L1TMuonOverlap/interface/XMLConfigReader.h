#ifndef OMTF_XMLConfigReader_H
#define OMTF_XMLConfigReader_H

#include <string>
#include <vector>
#include <ostream>
#include <memory>

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

  XMLConfigReader();
  ~XMLConfigReader();

  void readConfig(const std::string fName);

  void setConfigFile(const std::string & fName) {configFile = fName;}

  void setPatternsFile(const std::string & fName) {patternsFile = fName;}

  void setEventsFile(const std::string & fName) {eventsFile = fName;}

  std::vector<std::shared_ptr<GoldenPattern>> readPatterns(const L1TMuonOverlapParams &aConfig);

  void readLUTs(std::vector<l1t::LUT *> luts, const L1TMuonOverlapParams & aConfig, const std::vector<std::string> & types);

  void readConfig(L1TMuonOverlapParams *aConfig) const;

  unsigned int getPatternsVersion() const;

  std::vector<std::vector<int> > readEvent(unsigned int iEvent=0,
					   unsigned int iProcessor=0,
					   bool readEta = false);

 private:

  std::string configFile; //XML file with general configuration
  std::string patternsFile; //XML file with GoldenPatterns
  std::string eventsFile;   //XML file with events

  std::unique_ptr<GoldenPattern> buildGP(xercesc::DOMElement* aGPElement,
			  const L1TMuonOverlapParams & aConfig,
			  unsigned int index=0,
			  unsigned int aGPNumber=999);
  
  //  xercesc::XercesDOMParser *parser;
  //  xercesc::DOMDocument* doc;

  ///Cache with GPs read.
  std::vector<std::shared_ptr<GoldenPattern>> aGPs;

};

#endif
