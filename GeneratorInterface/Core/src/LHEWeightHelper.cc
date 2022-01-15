#include "GeneratorInterface/Core/interface/LHEWeightHelper.h"
#include <boost/algorithm/string/join.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <iostream>
#include <stdexcept>

using namespace tinyxml2;

namespace gen {
  void LHEWeightHelper::setHeaderLines(std::vector<std::string> headerLines) { 
    headerLines_ = headerLines;
  }

  void LHEWeightHelper::parseWeights() {
    parsedWeights_.clear();

    if (!isConsistent() && failIfInvalidXML_) {
      throw std::runtime_error(
          "XML in LHE is not consistent: Most likely, tags were swapped.\n"
          "To turn on fault fixing, use 'setFailIfInvalidXML(false)'\n"
          "WARNING: the tag swapping may lead to weights associated with the incorrect group");
    } else if (!isConsistent()) {
      swapHeaders();
    }

    tinyxml2::XMLDocument xmlDoc;
    std::string fullHeader = boost::algorithm::join(headerLines_, "");
    if (debug_)
      std::cout << "Full header is \n" << fullHeader << std::endl;

    int xmlError = xmlDoc.Parse(fullHeader.c_str());
    // in case of &gt; instead of <
    if (xmlError != 0) {
      boost::replace_all(fullHeader, "&lt;", "<");
      boost::replace_all(fullHeader, "&gt;", ">");
      xmlError = xmlDoc.Parse(fullHeader.c_str());
    }
    // error persists (how to handle error?)
    if (xmlError != 0) {
      std::cerr << "WARNING: Error in parsing XML of LHE weight header!" << std::endl;
      xmlDoc.PrintError();
      if (failIfInvalidXML_)
        throw std::runtime_error("XML is unreadable because of above error.");
      else
        return;
    }

    std::vector<std::string> nameAlts_ = {"name", "type"};

    int weightIndex = 0;
    int groupIndex = 0;
    for (auto* e = xmlDoc.RootElement(); e != nullptr; e = e->NextSiblingElement()) {
      if (debug_) 
        std::cout << "XML element is " << e->Name() << std::endl;
      std::string groupName = "";
      if (strcmp(e->Name(), "weight") == 0) {
        if (debug_) 
          std::cout << "Found weight unmatched to group\n";
        // we are here if there is a weight that does not belong to any group
        // TODO: Recylce code better between here when a weight is found in a group
        std::string text = "";
        if (e->GetText()) {
          text = e->GetText();
        }
        std::unordered_map<std::string, std::string> attributes;
        for (auto* att = e->FirstAttribute(); att != nullptr; att = att->Next())
          attributes[att->Name()] = att->Value();
        parsedWeights_.push_back({e->Attribute("id"), weightIndex++, groupName, text, attributes, groupIndex});
      } else if (strcmp(e->Name(), "weightgroup") == 0) {
        if (debug_)
          std::cout << "Found a weight group.\n";
        // to deal wiht files with "id" instead of "name"
        for (auto nameAtt : nameAlts_) {
          if (e->Attribute(nameAtt.c_str())) {
            groupName = e->Attribute(nameAtt.c_str());
            break;
          }
        }
        if (groupName.empty()) {
          // TODO: Need a better failure mode
          throw std::runtime_error("couldn't find groupname");
        }
        // May remove this, very specific error
        if (groupName.find(".") != std::string::npos)
          groupName.erase(groupName.find("."), groupName.size());

        for (auto* inner = e->FirstChildElement("weight"); inner != nullptr;
             inner = inner->NextSiblingElement("weight")) {
          // we are here if there is a weight in a weightgroup
          if (debug_)
            std::cout << "Found a weight inside the group. Content is " << inner->GetText() <<  " group index is " << groupIndex << std::endl;
          std::string text = "";
          if (inner->GetText())
            text = inner->GetText();
          std::unordered_map<std::string, std::string> attributes;
          for (auto* att = inner->FirstAttribute(); att != nullptr; att = att->Next())
            attributes[att->Name()] = att->Value();
          parsedWeights_.push_back({inner->Attribute("id"), weightIndex++, groupName, text, attributes, groupIndex});
        }
      } else
          std::cout << "Found an invalid entry\n";
      groupIndex++;
    }
    buildGroups();
  }

  bool LHEWeightHelper::isConsistent() {
    int curLevel = 0;

    for (auto line : headerLines_) {
      if (line.find("<weightgroup") != std::string::npos) {
        curLevel++;
        if (curLevel != 1) {
          return false;
        }
      } else if (line.find("</weightgroup>") != std::string::npos) {
        curLevel--;
        if (curLevel != 0) {
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
      if (line.find("<weightgroup") != std::string::npos) {
        curLevel++;
        if (curLevel != 1) {
          open = idx;
        }
      } else if (line.find("</weightgroup>") != std::string::npos) {
        curLevel--;
        if (curLevel != 0) {
          close = idx;
        }
      }
      if (open > -1 && close > -1) {
        std::swap(headerLines_[open], headerLines_[close]);
        open = -1;
        close = -1;
      }
    }
  }
}  // namespace gen
