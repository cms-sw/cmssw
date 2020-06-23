#ifndef DETECTOR_DESCRIPTION_FILTER_H
#define DETECTOR_DESCRIPTION_FILTER_H

// -*- C++ -*-
//
// Package:    DetectorDescription/Filter
// Class:      Filter
//
/**\class Filter

 Description: Filter list

 Implementation:
     Filter criteria is defined in XML
*/
//
// Original Author:  Ianna Osborne
//         Created:  Tue, 12 Mar 2019 09:51:33 CET
//
//
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <memory>
#include <vector>
#include <regex>

namespace cms {
  struct DDSpecPar;

  struct Filter {
    void print() const {
      edm::LogVerbatim("Geometry").log([&](auto& log) {
        for (const auto& it : skeys) {
          log << it << ", ";
        }
        if (next) {
          log << "Next:\n";
          print(next);
        }
        if (up) {
          log << "Up:\n";
          up->print();
        }
      });
    }

    void print(const std::unique_ptr<Filter>& filter) const {
      edm::LogVerbatim("Geometry").log([&](auto& log) {
        for (const auto& it : filter->skeys) {
          log << it << ", ";
        }
      });
    }

    std::vector<std::string_view> skeys;
    std::vector<std::regex> keys;
    std::unique_ptr<Filter> next;
    struct Filter* up;
    const DDSpecPar* spec = nullptr;
  };

  namespace dd {
    bool accepted(std::vector<std::regex> const&, std::string_view);
    bool isRegex(std::string_view);
    bool isMatch(std::string_view, std::string_view);
    bool compareEqual(std::string_view, std::string_view);
    bool compareEqual(std::string_view, std::regex);
    std::string_view realTopName(std::string_view);
    std::vector<std::string_view> split(std::string_view, const char*);
    std::string_view noNamespace(std::string_view);
  }  // namespace dd
}  // namespace cms

#endif
