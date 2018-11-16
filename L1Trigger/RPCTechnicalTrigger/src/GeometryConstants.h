#ifndef L1Trigger_RPCTechnicalTrigger_GeometryConstants_h
#define L1Trigger_RPCTechnicalTrigger_GeometryConstants_h
// -*- C++ -*-
//
// Package:     L1Trigger/RPCTechnicalTrigger
// Class  :     GeometryConstants
// 
/**\class GeometryConstants GeometryConstants.h "GeometryConstants.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Fri, 16 Nov 2018 19:59:59 GMT
//

// system include files
#include <array>
#include <map>
// user include files

// forward declarations
namespace rpctechnicaltrigger {
  constexpr std::array<int,5> s_wheelid = { {-2, -1, 0, 1, 2} };
  constexpr std::array<int,6> s_sec1id = { {12, 2, 4, 6, 8, 10} };
  constexpr std::array<int,6> s_sec2id = { {1, 3, 5, 7, 9, 11} };

  extern const std::map<int, int> s_layermap;
}


#endif
