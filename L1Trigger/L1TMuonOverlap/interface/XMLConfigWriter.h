#ifndef OMTF_XMLConfigWriter_H
#define OMTF_XMLConfigWriter_H

#include <string>
#include <vector>

#include "xercesc/util/XercesDefs.hpp"

#include "L1Trigger/L1TMuonOverlap/interface/OMTFConfiguration.h"
#include "L1Trigger/L1TMuonOverlap/interface/OmtfName.h"

class GoldenPattern;
class OMTFConfiguration;
class OMTFinput;
class OMTFResult;
class AlgoMuon;
namespace l1t {
  class RegionalMuonCand;
}
struct Key;

namespace XERCES_CPP_NAMESPACE {
  class DOMElement;
  class DOMDocument;
  class DOMImplementation;
}  // namespace XERCES_CPP_NAMESPACE

class XMLConfigWriter {
public:
  XMLConfigWriter(const OMTFConfiguration* aOMTFConfig);

  void initialiseXMLDocument(const std::string& docName);

  void finaliseXMLDocument(const std::string& fName);

  xercesc::DOMElement* writeEventHeader(unsigned int eventId, unsigned int mixedEventId = 0);

  xercesc::DOMElement* writeEventData(xercesc::DOMElement* aTopElement, const OmtfName& board, const OMTFinput& aInput);

  void writeAlgoMuon(xercesc::DOMElement* aTopElement, unsigned int iRefHit, const AlgoMuon& aMuon);

  void writeCandMuon(xercesc::DOMElement* aTopElement, const l1t::RegionalMuonCand& aCand);

  void writeResultsData(xercesc::DOMElement* aTopElement,
                        unsigned int iRegion,
                        const Key& aKey,
                        const OMTFResult& aResult);

  void writeGPData(const GoldenPattern& aGP);

  void writeGPData(const GoldenPattern& aGP1,
                   const GoldenPattern& aGP2,
                   const GoldenPattern& aGP3,
                   const GoldenPattern& aGP4);

  void writeConnectionsData(const std::vector<std::vector<OMTFConfiguration::vector2D> >& measurements4D);

  unsigned int findMaxInput(const OMTFConfiguration::vector1D& myCounts);

private:
  xercesc::DOMImplementation* domImpl;
  xercesc::DOMElement* theTopElement;
  xercesc::DOMDocument* theDoc;

  const OMTFConfiguration* myOMTFConfig;
};

//////////////////////////////////
//////////////////////////////////
#endif
