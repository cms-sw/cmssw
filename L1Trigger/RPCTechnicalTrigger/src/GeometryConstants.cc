// -*- C++ -*-
//
// Package:     L1Trigger/RPCTechnicalTrigger
// Class  :     GeometryConstants
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Fri, 16 Nov 2018 20:00:03 GMT
//

// system include files

// user include files
#include "GeometryConstants.h"

namespace rpctechnicaltrigger {
    const std::map<int, int> s_layermap = { {113,   0},  //RB1InFw
                                          {123,   1},  //RB1OutFw
  
                                          {20213, 2},  //RB22Fw
                                          {20223, 2},  //RB22Fw
                                          {30223, 3},  //RB23Fw
                                          {30213, 3},  //RB23Fw
                                          {30212, 4},  //RB23M
                                          {30222, 4},  //RB23M
  
                                          {313,   5},  //RB3Fw
                                          {413,   6},  //RB4Fw
                                          {111,   7},  //RB1InBk
                                          {121,   8},  //RB1OutBk
  
                                          {20211, 9},  //RB22Bw
                                          {20221, 9},  //RB22Bw
                                          {30211,10}, //RB23Bw
                                          {30221,10}, //RB23Bw
  
                                          {311,  11}, //RB3Bk
                                          {411,  12} //RB4Bk
    };
}
