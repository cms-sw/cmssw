#include "GeneratorInterface/Core/interface/LHEWeightHelper.h"
#include <boost/algorithm/string/join.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <iostream>
#include <stdexcept>

using namespace tinyxml2;

namespace gen {
  void LHEWeightHelper::setHeaderLines(std::vector<std::string> headerLines) { headerLines_ = headerLines; }

  bool LHEWeightHelper::parseLHE(tinyxml2::XMLDocument& xmlDoc) {
    parsedWeights_.clear();

    std::string fullHeader = boost::algorithm::join(headerLines_, "");

    if (debug_)
      std::cout << "Full header is \n" << fullHeader << std::endl;
    int xmlError = xmlDoc.Parse(fullHeader.c_str());
    ErrorType errorType;

    while (errorType = findErrorType(xmlError, fullHeader), errorType != ErrorType::NoError) {
      if (failIfInvalidXML_) {
        xmlDoc.PrintError();
        throw cms::Exception("LHEWeightHelper")
            << "The LHE header is not valid XML! Weight information was not properly parsed.";
      } else if (errorType == ErrorType::HTMLStyle) {
        if (debug_)
          std::cout << "  >>> This file uses &gt; instead of >\n";
        xmlError = tryReplaceHtmlStyle(xmlDoc, fullHeader);
      } else if (errorType == ErrorType::SwapHeader) {
        if (debug_)
          std::cout << "  >>> Some headers in the file are swapped\n";
        swapHeaders();
        fullHeader = boost::algorithm::join(headerLines_, "");
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

  void LHEWeightHelper::addGroup(tinyxml2::XMLElement* inner, std::string groupName, int groupIndex, int& weightIndex) {
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
    parsedWeights_.push_back({inner->Attribute("id"), weightIndex++, groupName, text, attributes, groupIndex});
  }

  void LHEWeightHelper::parseWeights() {
    tinyxml2::XMLDocument xmlDoc;
    if (!parseLHE(xmlDoc)) {
      return;
    }

    int weightIndex = 0;
    int groupIndex = 0;
    for (auto* e = xmlDoc.RootElement(); e != nullptr; e = e->NextSiblingElement()) {
      if (debug_)
        std::cout << "XML element is " << e->Name() << std::endl;
      std::string groupName = "";
      if (strcmp(e->Name(), "weight") == 0) {
        if (debug_)
          std::cout << "Found weight unmatched to group\n";
        addGroup(e, groupName, groupIndex, weightIndex);
      } else if (strcmp(e->Name(), "weightgroup") == 0) {
        groupName = parseGroupName(e);
        if (debug_)
          std::cout << ">>>> Found a weight group: " << groupName << std::endl;
        for (auto inner = e->FirstChildElement("weight"); inner != nullptr; inner = inner->NextSiblingElement("weight"))
          addGroup(inner, groupName, groupIndex, weightIndex);
      }
      groupIndex++;
    }
    buildGroups();
    if (debug_)
      printWeights();
  }

  std::string LHEWeightHelper::parseGroupName(tinyxml2::XMLElement* el) {
    std::vector<std::string> nameAlts_ = {"name", "type"};
    for (const auto& nameAtt : nameAlts_) {
      if (el->Attribute(nameAtt.c_str())) {
        std::string groupName = el->Attribute(nameAtt.c_str());
        if (groupName.find('.') != std::string::npos)
          groupName.erase(groupName.find('.'), groupName.size());
        return groupName;
      }
    }

    if (!fillEmptyIfWeightFails_) {
      throw std::runtime_error("couldn't find groupname");
    }
    return "";
  }

  bool LHEWeightHelper::isConsistent() {
    int curLevel = 0;

    for (const auto& line : headerLines_) {
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

  void LHEWeightHelper::swapHeaders() {
    int curLevel = 0;
    int open = -1;
    int close = -1;
    for (size_t idx = 0; idx < headerLines_.size(); idx++) {
      std::string line = headerLines_[idx];
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
        std::swap(headerLines_[open], headerLines_[close]);
        open = -1;
        close = -1;
      }
    }
  }

  tinyxml2::XMLError LHEWeightHelper::tryReplaceHtmlStyle(tinyxml2::XMLDocument& xmlDoc, std::string& fullHeader) {
    // in case of &gt; instead of <
    boost::replace_all(fullHeader, "&lt;", "<");
    boost::replace_all(fullHeader, "&gt;", ">");

    return xmlDoc.Parse(fullHeader.c_str());
  }

  tinyxml2::XMLError LHEWeightHelper::tryRemoveTrailings(tinyxml2::XMLDocument& xmlDoc, std::string& fullHeader) {
    // delete extra strings after the last </weightgroup> (occasionally contain '<' or '>')
    std::size_t theLastKet = fullHeader.rfind(weightgroupKet_) + weightgroupKet_.length();
    std::size_t thelastWeight = fullHeader.rfind(weightTag_) + weightTag_.length();
    fullHeader = fullHeader.substr(0, std::max(theLastKet, thelastWeight));

    return xmlDoc.Parse(fullHeader.c_str());
  }

  LHEWeightHelper::ErrorType LHEWeightHelper::findErrorType(int xmlError, std::string& fullHeader) {
    if (fullHeader.size() == 0)
      return ErrorType::Empty;
    else if (!isConsistent())
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
    } else
      return ErrorType::NoError;
  }
}  // namespace gen
