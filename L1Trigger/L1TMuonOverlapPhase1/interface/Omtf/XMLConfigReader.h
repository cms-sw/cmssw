#ifndef L1T_OmtfP1_XMLConfigReader_H
#define L1T_OmtfP1_XMLConfigReader_H

#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OMTFConfiguration.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/GoldenPatternBase.h"

#include "CondFormats/L1TObjects/interface/LUT.h"

#include <string>
#include <vector>
#include <ostream>
#include <memory>

#include "xercesc/util/XercesDefs.hpp"
#include "xercesc/dom/DOM.hpp"

class GoldenPattern;
class OMTFConfiguration;
class L1TMuonOverlapParams;

namespace XERCES_CPP_NAMESPACE {

  class DOMElement;
  class XercesDOMParser;

}  // namespace XERCES_CPP_NAMESPACE

class XMLConfigReader {
public:
  XMLConfigReader();
  ~XMLConfigReader();

  void readConfig(const std::string fName);

  void setConfigFile(const std::string& fName) { configFile = fName; }

  void setPatternsFiles(std::vector<std::string>& fNames) { patternsFiles = fNames; }

  void setEventsFile(const std::string& fName) { eventsFile = fName; }

  /*
   * if buildEmptyPatterns == true - the patterns with the thePt = 0 are added to the to the output vector of the goledePatterns
   * it is needed in the readLUTs, as only with the empty patterns it is possible to obtain the patterns grouping from the LUTs in the L1TMuonOverlapParams
   */
  template <class GoldenPatternType>
  GoldenPatternVec<GoldenPatternType> readPatterns(const L1TMuonOverlapParams& aConfig,
                                                   const std::string& patternsFile,
                                                   bool buildEmptyPatterns,
                                                   bool resetNumbering = true);

  template <class GoldenPatternType>
  GoldenPatternVec<GoldenPatternType> readPatterns(const L1TMuonOverlapParams& aConfig,
                                                   const std::vector<std::string>& patternsFiles,
                                                   bool buildEmptyPatterns);

  void readLUTs(std::vector<l1t::LUT*> luts,
                const L1TMuonOverlapParams& aConfig,
                const std::vector<std::string>& types);

  void readConfig(L1TMuonOverlapParams* aConfig) const;

  unsigned int getPatternsVersion() const;

  std::vector<std::vector<int> > readEvent(unsigned int iEvent = 0, unsigned int iProcessor = 0, bool readEta = false);

private:
  std::string configFile;                  //XML file with general configuration
  std::vector<std::string> patternsFiles;  //XML files with GoldenPatterns
  std::string eventsFile;                  //XML file with events

  template <class GoldenPatternType>
  std::unique_ptr<GoldenPatternType> buildGP(xercesc::DOMElement* aGPElement,
                                             const L1TMuonOverlapParams& aConfig,
                                             unsigned int patternGroup,
                                             unsigned int index = 0,
                                             unsigned int aGPNumber = 999);

  unsigned int iGPNumber = 0;
  unsigned int iPatternGroup = 0;
};

#endif
