#include "GeneratorInterface/Core/interface/LHEWeightHelper.h"
#include <boost/algorithm/string/join.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <iostream>
#include <stdexcept>

using namespace tinyxml2;

namespace gen {
  bool LHEWeightHelper::validateAndFixHeader(std::vector<std::string>& headerLines,
                                             tinyxml2::XMLDocument& xmlDoc) const {
    std::string fullHeader = boost::algorithm::join(headerLines, "");

    if (debug_)
      std::cout << "Full header is \n" << fullHeader << std::endl;
    int xmlError = xmlDoc.Parse(fullHeader.c_str());
    ErrorType errorType;

    while (errorType = findErrorType(xmlError, fullHeader), errorType != ErrorType::NoError) {
      if (failIfInvalidXML_) {
        std::cout << "XML error: ";
        xmlDoc.PrintError();
        throw cms::Exception("LHEWeightHelper")
            << "The LHE header is not valid! Weight information was not properly parsed."
            << " The error type is '" << errorTypeAsString_.at(errorType) << "'";
      } else if (errorType == ErrorType::HTMLStyle) {
        if (debug_)
          std::cout << "  >>> This file uses &gt; instead of >\n";
        xmlError = tryReplaceHtmlStyle(xmlDoc, fullHeader);
      } else if (errorType == ErrorType::SwapHeader) {
        if (debug_)
          std::cout << "  >>> Some headers in the file are swapped\n";
        std::vector<std::string> fixedHeaderLines;
        boost::split(fixedHeaderLines, fullHeader, boost::is_any_of("\n"));
        swapHeaders(fixedHeaderLines);
        fullHeader = boost::algorithm::join(fixedHeaderLines, "\n");
        xmlError = xmlDoc.Parse(fullHeader.c_str());
      } else if (errorType == ErrorType::TrailingStr) {
        if (debug_)
          std::cout << "  >>> There is non-XML text at the end of this file\n";
        xmlError = tryRemoveTrailings(xmlDoc, fullHeader);
      } else if (errorType == ErrorType::Empty) {
        if (debug_)
          std::cout << "  >>> There are no LHE xml header, ending parsing" << std::endl;
        return false;
      } else if (errorType == ErrorType::NoWeightGroup) {
        if (debug_)
          std::cout << "  >>> There are no <weightgroup> in the LHE xml header, ending parsing" << std::endl;
        return false;
      } else {
        std::string error = "Fatal error when parsing the LHE header. The header is not valid XML! Parsing error was ";
        error += xmlDoc.ErrorStr();
        throw cms::Exception("LHEWeightHelper") << error;
      }
    }
    return true;
  }

  ParsedWeight LHEWeightHelper::parseWeight(tinyxml2::XMLElement* inner,
                                            std::string groupName,
                                            int groupIndex,
                                            int& weightIndex) const {
    if (debug_)
      std::cout << "  >> Found a weight inside the group. " << std::endl;
    std::string text = "";
    if (inner->GetText())
      text = inner->GetText();

    std::unordered_map<std::string, std::string> attributes;
    for (auto* att = inner->FirstAttribute(); att != nullptr; att = att->Next())
      attributes[att->Name()] = att->Value();
    if (debug_)
      std::cout << "     " << weightIndex << ": \"" << text << "\"" << std::endl;
    return {inner->Attribute("id"), weightIndex++, groupName, text, attributes, groupIndex};
  }

  std::vector<std::unique_ptr<gen::WeightGroupInfo>> LHEWeightHelper::parseWeights(std::vector<std::string> headerLines,
                                                                                   bool addUnassociatedGroup) const {
    tinyxml2::XMLDocument xmlDoc;
    if (!validateAndFixHeader(headerLines, xmlDoc)) {
      return {};
    }

    std::vector<ParsedWeight> parsedWeights;
    int weightIndex = 0;
    int groupIndex = 0;
    for (auto* e = xmlDoc.RootElement(); e != nullptr; e = e->NextSiblingElement()) {
      if (debug_)
        std::cout << "XML element is " << e->Name() << std::endl;
      std::string groupName = "";
      if (strcmp(e->Name(), "weight") == 0) {
        if (debug_)
          std::cout << "Found weight unmatched to group\n";
        parsedWeights.push_back(parseWeight(e, groupName, groupIndex, weightIndex));
      } else if (strcmp(e->Name(), "weightgroup") == 0) {
        groupName = parseGroupName(e);
        if (debug_)
          std::cout << ">>>> Found a weight group: " << groupName << std::endl;
        for (auto inner = e->FirstChildElement("weight"); inner != nullptr; inner = inner->NextSiblingElement("weight"))
          parsedWeights.push_back(parseWeight(inner, groupName, groupIndex, weightIndex));
      }
      groupIndex++;
    }
    auto groups = buildGroups(parsedWeights, addUnassociatedGroup);
    if (debug_)
      printWeights(groups);
    return groups;
  }

  std::string LHEWeightHelper::parseGroupName(tinyxml2::XMLElement* el) const {
    std::vector<std::string> nameAlts = {"name", "type"};
    for (const auto& nameAtt : nameAlts) {
      if (el->Attribute(nameAtt.c_str())) {
        std::string groupName = el->Attribute(nameAtt.c_str());
        if (groupName.find('.') != std::string::npos)
          groupName.erase(groupName.find('.'), groupName.size());
        return groupName;
      }
    }

    throw cms::Exception("LHEWeightHelper") << "Could not parse a name for weight group";
    return "";
  }

  bool LHEWeightHelper::isConsistent(const std::string& fullHeader) const {
    std::vector<std::string> headerLines;
    boost::split(headerLines, fullHeader, boost::is_any_of("\n"));
    int curLevel = 0;

    for (const auto& line : headerLines) {
      if (line.find("/weightgroup") != std::string::npos) {
        curLevel--;
        if (curLevel != 0) {
          return false;
        }
      } else if (line.find("weightgroup") != std::string::npos) {
        curLevel++;
        if (curLevel != 1) {
          return false;
        }
      }
    }
    return curLevel == 0;
  }

  void LHEWeightHelper::swapHeaders(std::vector<std::string>& headerLines) const {
    int curLevel = 0;
    int open = -1;
    int close = -1;
    for (size_t idx = 0; idx < headerLines.size(); idx++) {
      std::string& line = headerLines[idx];
      std::cout << "Line is " << line << std::endl;
      ;
      if (line.find("/weightgroup") != std::string::npos) {
        curLevel--;
        if (curLevel != 0) {
          close = idx;
        }
      } else if (line.find("weightgroup") != std::string::npos) {
        curLevel++;
        if (curLevel != 1) {
          open = idx;
        }
      }
      if (open > -1 && close > -1) {
        std::swap(headerLines[open], headerLines[close]);
        open = -1;
        close = -1;
      }
    }
  }

  tinyxml2::XMLError LHEWeightHelper::tryReplaceHtmlStyle(tinyxml2::XMLDocument& xmlDoc,
                                                          std::string& fullHeader) const {
    // in case of &gt; instead of <
    boost::replace_all(fullHeader, "&lt;", "<");
    boost::replace_all(fullHeader, "&gt;", ">");

    return xmlDoc.Parse(fullHeader.c_str());
  }

  tinyxml2::XMLError LHEWeightHelper::tryRemoveTrailings(tinyxml2::XMLDocument& xmlDoc, std::string& fullHeader) const {
    // delete extra strings after the last </weightgroup> (occasionally contain '<' or '>')
    std::size_t theLastKet = fullHeader.rfind(weightgroupKet_) + weightgroupKet_.length();
    std::size_t thelastWeight = fullHeader.rfind(weightTag_) + weightTag_.length();
    fullHeader = fullHeader.substr(0, std::max(theLastKet, thelastWeight));

    return xmlDoc.Parse(fullHeader.c_str());
  }

  LHEWeightHelper::ErrorType LHEWeightHelper::findErrorType(int xmlError, const std::string& fullHeader) const {
    if (fullHeader.empty())
      return ErrorType::Empty;
    else if (!isConsistent(fullHeader))
      return ErrorType::SwapHeader;
    else if (fullHeader.find("&lt;") != std::string::npos || fullHeader.find("&gt;") != std::string::npos)
      return ErrorType::HTMLStyle;
    else if (xmlError != 0) {
      if (fullHeader.rfind(weightgroupKet_) == std::string::npos)
        return ErrorType::NoWeightGroup;
      std::string trailingCand =
          fullHeader.substr(fullHeader.rfind(weightgroupKet_) + std::string(weightgroupKet_).length());
      if (trailingCand.find('<') == std::string::npos || trailingCand.find('>') == std::string::npos)
        return ErrorType::TrailingStr;
      else
        return ErrorType::Unknown;
    }
    return ErrorType::NoError;
  }
}  // namespace gen
