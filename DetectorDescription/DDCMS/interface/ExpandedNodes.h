#ifndef DETECTOR_DESCRIPTION_EXPANDED_NODES_H
#define DETECTOR_DESCRIPTION_EXPANDED_NODES_H

// -*- C++ -*-
//
// Package:    DetectorDescription/ExpandedNodes
// Class:      ExpandedNodes
//
/**\class ExpandedNodes

 Description: ExpandedNodes extra attributes: tags, offsets
              and copy numbers

 Implementation:
     ExpandedNodes structure to keep the nodes history
*/
//
// Original Author:  Ianna Osborne
//         Created:  Tue, 18 Mar 2019 12:22:35 CET
//
//
#include <vector>

namespace cms {

  struct ExpandedNodes {
    std::vector<double> tags;
    std::vector<double> offsets;
    std::vector<int> copyNos;
  };
}  // namespace cms

#endif
