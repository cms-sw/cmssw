#ifndef GeneratorInterface_Core_LHEWeightHelper_h
#define GeneratorInterface_Core_LHEWeightHelper_h

#include <string>
#include <vector>
#include <map>
#include <regex>
#include <fstream>

#include "SimDataFormats/GeneratorProducts/interface/UnknownWeightGroupInfo.h"
#include "SimDataFormats/GeneratorProducts/interface/MEParamWeightGroupInfo.h"
#include "SimDataFormats/GeneratorProducts/interface/PdfWeightGroupInfo.h"
#include "SimDataFormats/GeneratorProducts/interface/ScaleWeightGroupInfo.h"
#include "SimDataFormats/GeneratorProducts/interface/LHERunInfoProduct.h"
#include "GeneratorInterface/Core/interface/WeightHelper.h"

#include <tinyxml2.h>

namespace gen {
  class LHEWeightHelper : public WeightHelper {
  public:
    LHEWeightHelper() : WeightHelper(){};

    enum class ErrorType { Empty, SwapHeader, HTMLStyle, NoWeightGroup, TrailingStr, Unknown, NoError };

    void setHeaderLines(std::vector<std::string> headerLines);
    void parseWeights();
    bool isConsistent();
    void swapHeaders();
    void setFailIfInvalidXML(bool value) { failIfInvalidXML_ = value; }

  private:
    std::vector<std::string> headerLines_;
    std::string weightgroupKet_ = "</weightgroup>";
    std::string weightTag_ = "</weight>";
    bool failIfInvalidXML_ = false;
    std::string parseGroupName(tinyxml2::XMLElement* el);
    void addGroup(tinyxml2::XMLElement* inner, std::string groupName, int groupIndex, int& weightIndex);
    bool parseLHE(tinyxml2::XMLDocument& xmlDoc);
    tinyxml2::XMLError tryReplaceHtmlStyle(tinyxml2::XMLDocument& xmlDoc, std::string& fullHeader);
    tinyxml2::XMLError tryRemoveTrailings(tinyxml2::XMLDocument& xmlDoc, std::string& fullHeader);
    ErrorType findErrorType(int xmlError, std::string& fullHeader);
  };
}  // namespace gen

#endif
