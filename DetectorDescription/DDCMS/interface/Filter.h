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
#include <vector>

namespace cms {
  
  struct Filter {
    std::vector<std::string_view> keys;
    struct Filter* next;
  };
}

#endif
