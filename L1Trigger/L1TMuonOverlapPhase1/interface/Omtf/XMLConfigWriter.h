#ifndef OMTF_XMLConfigWriter_H
#define OMTF_XMLConfigWriter_H

#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OmtfName.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OMTFConfiguration.h"
#include <string>
#include <vector>

#include "xercesc/util/XercesDefs.hpp"

class GoldenPattern;
class OMTFConfiguration;
class OMTFinput;
class GoldenPatternResult;
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
  XMLConfigWriter(const OMTFConfiguration* aOMTFConfig,
                  bool writePdfThresholds = false,
                  bool writeMeanDistPhi1 = false);

  void initialiseXMLDocument(const std::string& docName);

  void finaliseXMLDocument(const std::string& fName);

  xercesc::DOMElement* writeEventHeader(unsigned int eventId, unsigned int mixedEventId = 0);

  xercesc::DOMElement* writeEventData(xercesc::DOMElement* aTopElement, const OmtfName& board, const OMTFinput& aInput);

  void writeAlgoMuon(xercesc::DOMElement* aTopElement, const AlgoMuon& aMuon);

  void writeCandMuon(xercesc::DOMElement* aTopElement, const l1t::RegionalMuonCand& aCand);

  void writeResultsData(xercesc::DOMElement* aTopElement,
                        unsigned int iRegion,
                        const Key& aKey,
                        const GoldenPatternResult& aResult);

  void writeGPData(GoldenPattern& aGP);

  void writeGPData(GoldenPattern* aGP1, GoldenPattern* aGP2, GoldenPattern* aGP3, GoldenPattern* aGP4);

  template <class GoldenPatternType>
  void writeGPs(const std::vector<std::shared_ptr<GoldenPatternType> >& goldenPats, std::string fName);

  void writeConnectionsData(const std::vector<std::vector<OMTFConfiguration::vector2D> >& measurements4D);

  unsigned int findMaxInput(const OMTFConfiguration::vector1D& myCounts);

private:
  xercesc::DOMImplementation* domImpl;
  xercesc::DOMElement* theTopElement;
  xercesc::DOMDocument* theDoc;

  const OMTFConfiguration* myOMTFConfig;

  bool writePdfThresholds = false;
  bool writeMeanDistPhi1 = false;
};

//////////////////////////////////
//////////////////////////////////
#endif
