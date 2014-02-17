// -*- C++ -*-
//
// Package:     RPCObjects
// Class  :     L1RPCHwConfig
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Wed Apr  9 13:48:10 CEST 2008
// $Id: L1RPCHwConfig.cc,v 1.1 2008/04/09 15:16:53 fruboes Exp $
//

// system include files

// user include files
#include "CondFormats/RPCObjects/interface/L1RPCHwConfig.h"

#include <iostream>
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
L1RPCHwConfig::L1RPCHwConfig()  
{
}

// L1RPCHwConfig::L1RPCHwConfig(const L1RPCHwConfig& rhs)
// {
//    // do actual copying here;
// }

L1RPCHwConfig::~L1RPCHwConfig()
{
}


void  L1RPCHwConfig::enablePAC(int tower, int sector, int segment, bool enable) 
{

 if (enable){
//  std::cout << "+";
  m_disabledDevices.erase(L1RPCDevCoords(tower, sector, segment)) ; 
 }
 else{
 // std::cout << "-";
  m_disabledDevices.insert(L1RPCDevCoords(tower, sector, segment)) ; 
 }

}

void L1RPCHwConfig::enableTower(int tower, bool enable) {


  for (int sec = 0; sec <12 ; ++sec  ){
    for (int seg = 0; seg<12; ++seg ) {
     enablePAC(tower,sec,seg,enable);

    }

  }


}

void L1RPCHwConfig::enableTowerInCrate(int tower, int crate, bool enable){
    for (int seg = 0; seg<12; ++seg ) {
      enablePAC(tower,crate,seg,enable);
    }
}

void L1RPCHwConfig::enableCrate(int crate, bool enable) {

  for (int tower = -16; tower < 17; ++tower){
     for (int seg = 0; seg<12; ++seg ) {
         enablePAC(tower, crate, seg, enable);
     }
  }


}


void L1RPCHwConfig::enableAll(bool enable){

   for (int seg = 0; seg<12; ++seg ) {
  //     std::cout <<  seg << " ";
       enableCrate(seg,enable);
  //     std::cout << std::endl; 
   }


}
