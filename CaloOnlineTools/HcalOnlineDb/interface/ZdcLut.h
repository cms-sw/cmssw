#ifndef CaloOnlineTools_HcalOnlineDb_ZdcLut_h
#define CaloOnlineTools_HcalOnlineDb_ZdcLut_h
// -*- C++ -*-
//
// Package:     HcalOnlineDb
// Class  :     ZdcLut
// 
/**\class ZdcLut ZdcLut.h CaloOnlineTools/HcalOnlineDb/interface/ZdcLut.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Elijah Dunn
//         Created:  Thu Aug 20 13:10:08 CEST 2009
// $Id$
//

#include <iostream>
#include <vector>
#include <string>

using namespace std;


/* definition of the 18 ZDC channels:
   DetId section side channel
13415278 ZDC_EM  0 1
13415279 ZDC_EM  0 2
13415280 ZDC_EM  0 3
13415281 ZDC_EM  0 4
13415282 ZDC_EM  0 5
13415269 ZDC_EM  1 1
13415270 ZDC_EM  1 2
13415271 ZDC_EM  1 3
13415272 ZDC_EM  1 4
13415273 ZDC_EM  1 5
13415283 ZDC_HAD 0 1
13415284 ZDC_HAD 0 2
13415285 ZDC_HAD 0 3
13415286 ZDC_HAD 0 4
13415274 ZDC_HAD 1 1
13415275 ZDC_HAD 1 2
13415276 ZDC_HAD 1 3
13415277 ZDC_HAD 1 4
*/

class ZdcLut
{

   public:
      ZdcLut();
      virtual ~ZdcLut();

      struct ZDC_channels {vector <int> LUT;};
      
      struct ZDC_fibers {vector <ZDC_channels> channel;};
      
      struct ZDC_sides {vector<ZDC_fibers> fiber;};

      //get_lut returns a specific lut based on side, fiber, and fiber_channel
      vector <int> get_lut(int side_num,
			   int fiber_num,
			   int channel_num){ return side[side_num].fiber[fiber_num].channel[channel_num].LUT; }

      vector<int> getLut(std::string zdc_section,
			 int zdc_zside,
			 int zdc_channel);
      
      int test(void);
      
      //variable
      vector <ZDC_sides> side;

   private:

};


#endif
