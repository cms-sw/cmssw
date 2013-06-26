#ifndef SiStripDetId_SiStripSubStructure_h
#define SiStripDetId_SiStripSubStructure_h
// -*- C++ -*-
//
// Package:     SiStripDetId
// Class  :     SiStripSubStructure
// 
/**\class SiStripSubStructure SiStripSubStructure.h DataFormats/SiStripDetId/interface/SiStripSubStructure.h

 Description: <Assign detector Ids to different substructures of the SiStripTracker: TOB, TIB, etc>

 Usage:
    <usage>

*/
//
// Original Author:  dkcira
//         Created:  Wed Jan 25 07:18:21 CET 2006
// $Id: SiStripSubStructure.h,v 1.6 2012/01/10 11:36:21 innocent Exp $
//

#include <vector>
#include <stdint.h>

class SiStripSubStructure
{

   public:
      SiStripSubStructure();
      ~SiStripSubStructure();

      void getTIBDetectors(const std::vector<uint32_t> & inputDetRawIds, // INPUT
                           std::vector<uint32_t> & tibDetRawIds,         // OUTPUT
                           uint32_t layer         = 0,                   // SELECTION: layer = 1..4, 0(ALL)
                           uint32_t bkw_frw = 0,                         // bkw_frw = 1(TIB-), 2(TIB+) 0(ALL)
                           uint32_t int_ext = 0,                         // int_ext = 1 (internal), 2(external), 0(ALL)
                           uint32_t string        = 0) const;            // string = 1..N, 0(ALL)

      void getTIDDetectors(const std::vector<uint32_t> & inputDetRawIds, // INPUT
                           std::vector<uint32_t> & tidDetRawIds,         // OUTPUT
                           uint32_t side        = 0,                     // SELECTION: side = 1(back), 2(front), 0(ALL)
                           uint32_t wheel       = 0,                     // wheel = 1..3, 0(ALL)
                           uint32_t ring        = 0,                     // ring  = 1..3, 0(ALL)
                           uint32_t ster        = 0) const;              // ster = 1(stereo), else (nonstereo), 0(ALL)

      void getTOBDetectors(const std::vector<uint32_t> & inputDetRawIds, // INPUT
                           std::vector<uint32_t> & tobDetRawIds,         // OUTPUT
                           uint32_t layer      = 0,                      // SELECTION: layer = 1..6, 0(ALL)
                           uint32_t bkw_frw    = 0,                      // bkw_frw = 1(TOB-) 2(TOB+) 0(everything)
                           uint32_t rod        = 0) const;               // rod = 1..N, 0(ALL)


      void getTECDetectors(const std::vector<uint32_t> & inputDetRawIds, // INPUT
                           std::vector<uint32_t> & tecDetRawIds,         // OUTPUT
                           uint32_t side          = 0,                   // SELECTION: side = 1(TEC-), 2(TEC+),  0(ALL)
                           uint32_t wheel         = 0,                   // wheel = 1..9, 0(ALL)
                           uint32_t petal_bkw_frw = 0,                   // petal_bkw_frw = 1(backward) 2(forward) 0(all)
                           uint32_t petal         = 0,                   // petal = 1..8, 0(ALL)
                           uint32_t ring          = 0,                   // ring = 1..7, 0(ALL)
                           uint32_t ster          = 0) const;            // ster = 1(sterero), else(nonstereo), 0(ALL)

   private:
      SiStripSubStructure(const SiStripSubStructure&); // stop default

      const SiStripSubStructure& operator=(const SiStripSubStructure&); // stop default

      // ---------- member data --------------------------------

};

#endif
