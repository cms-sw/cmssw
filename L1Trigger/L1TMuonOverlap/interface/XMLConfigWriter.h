#ifndef OMTF_XMLConfigWriter_H
#define OMTF_XMLConfigWriter_H

#include <string>
#include <vector>

#include "xercesc/util/XercesDefs.hpp"

#include "L1Trigger/L1TMuonOverlap/interface/OMTFConfiguration.h"

class GoldenPattern;
class OMTFConfiguration;
class OMTFinput;
class OMTFResult;
class InternalObj;
struct Key;

namespace XERCES_CPP_NAMESPACE{
  class DOMElement;
  class DOMDocument;
  class DOMImplementation;
}

class XMLConfigWriter{

 public:

  XMLConfigWriter();

  void initialiseXMLDocument(const std::string & docName);

  void finaliseXMLDocument(const std::string & fName);

  xercesc::DOMElement * writeEventHeader(unsigned int eventId,
					 unsigned int mixedEventId = 0);

  xercesc::DOMElement * writeEventData(xercesc::DOMElement *aTopElement,
				       unsigned int iProcessor,
				       const OMTFinput & aInput);

  void writeCandidateData(xercesc::DOMElement *aTopElement,
			  unsigned int iRefHit,
			  const InternalObj & aCand);

  void writeResultsData(xercesc::DOMElement *aTopElement,
			unsigned int iRegion,
			const Key& aKey,
			const OMTFResult & aResult);

  void writeGPData(const GoldenPattern & aGP);

  void writeGPData(const GoldenPattern & aGP1,
		   const GoldenPattern & aGP2);

  void writeGPData(const GoldenPattern & aGP1,
		   const GoldenPattern & aGP2,
		   const GoldenPattern & aGP3,
		   const GoldenPattern & aGP4);
		   
  void writeConnectionsData(const std::vector<std::vector <OMTFConfiguration::vector2D> > & measurements4D);

  unsigned int findMaxInput(const OMTFConfiguration::vector1D & myCounts);

 private:

  ///Utility function to set which layers should be ignored
  bool removeLayers(unsigned int iLayer);

  xercesc::DOMImplementation* domImpl;
  xercesc::DOMElement* theTopElement;
  xercesc::DOMDocument* theDoc;

};


//////////////////////////////////
//////////////////////////////////
#endif
