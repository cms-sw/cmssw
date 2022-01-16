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
    const std::unordered_map<ErrorType, std::string> errorTypeAsString_ = {
        {ErrorType::Empty, "Empty header"},
        {ErrorType::SwapHeader, "Header info out of order"},
        {ErrorType::HTMLStyle, "Header is invalid HTML"},
        {ErrorType::TrailingStr, "Header has extraneous info"},
        {ErrorType::Unknown, "Unregonized error"},
        {ErrorType::NoError, "No error here!"}};

    std::vector<std::unique_ptr<gen::WeightGroupInfo>> parseWeights(std::vector<std::string> headerLines,
                                                                    bool addUnassociated) const;
    bool isConsistent(const std::string& fullHeader) const;
    void swapHeaders(std::vector<std::string>& headerLines) const;
    void setFailIfInvalidXML(bool value) { failIfInvalidXML_ = value; }
    bool failIfInvalidXML() const { return failIfInvalidXML_; }

  private:
    std::string weightgroupKet_ = "</weightgroup>";
    std::string weightTag_ = "</weight>";
    bool failIfInvalidXML_ = false;
    std::string parseGroupName(tinyxml2::XMLElement* el) const;
    ParsedWeight parseWeight(tinyxml2::XMLElement* inner, std::string groupName, int groupIndex, int& weightIndex) const;
    bool validateAndFixHeader(std::vector<std::string>& headerLines, tinyxml2::XMLDocument& xmlDoc) const;
    tinyxml2::XMLError tryReplaceHtmlStyle(tinyxml2::XMLDocument& xmlDoc, std::string& fullHeader) const;
    tinyxml2::XMLError tryRemoveTrailings(tinyxml2::XMLDocument& xmlDoc, std::string& fullHeader) const;
    ErrorType findErrorType(int xmlError, const std::string& headerLines) const;
  };
}  // namespace gen

#endif
