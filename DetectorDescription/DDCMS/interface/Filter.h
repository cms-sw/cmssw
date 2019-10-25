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
#include <memory>
#include <vector>

namespace cms {
  struct DDSpecPar;

  struct Filter {
    std::vector<std::string_view> keys;
    std::unique_ptr<Filter> next;
    struct Filter* up;
    const DDSpecPar* spec = nullptr;
  };

  namespace dd {
    bool accepted(std::vector<std::string_view> const&, std::string_view);
    int contains(std::string_view, std::string_view);
    bool isRegex(std::string_view);
    bool compareEqual(std::string_view, std::string_view);
    std::string_view realTopName(std::string_view);
    std::vector<std::string_view> split(std::string_view, const char*);
    std::string_view noNamespace(std::string_view);
  }  // namespace dd
}  // namespace cms

#endif
