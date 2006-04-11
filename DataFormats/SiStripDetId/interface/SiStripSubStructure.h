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
// $Id: SiStripSubStructure.h,v 1.1 2006/02/09 18:53:33 gbruno Exp $
//

#include <boost/cstdint.hpp>
#include <vector>

class SiStripSubStructure
{

   public:
      SiStripSubStructure();
      virtual ~SiStripSubStructure();

      void getTIBDetectors(const std::vector<uint32_t> & inputDetRawIds, // input
                           std::vector<uint32_t> & tibDetRawIds,         // output
                           uint32_t layer         = 0,           // selection
                           uint32_t bkw_frw = 0,
                           uint32_t int_ext = 0,
                           uint32_t string        = 0) const;

      void getTIDDetectors(const std::vector<uint32_t> & inputDetRawIds, // input
                           std::vector<uint32_t> & tidDetRawIds,         // output
                           uint32_t side        = 0,                     // selection
                           uint32_t wheel       = 0,
                           uint32_t ring        = 0,
                           uint32_t ster        = 0) const;

      void getTOBDetectors(const std::vector<uint32_t> & inputDetRawIds, // input
                           std::vector<uint32_t> & tobDetRawIds,         // output
                           uint32_t layer      = 0,                      // selection
                           uint32_t bkw_frw    = 0,        // bkw_frw = 1(backward) 2(forward) 0(everything)
                           uint32_t rod        = 0) const;



      void getTECDetectors(const std::vector<uint32_t> & inputDetRawIds, // input
                           std::vector<uint32_t> & tecDetRawIds,         // output
                           uint32_t side          = 0,                     // selection
                           uint32_t wheel         = 0,
                           uint32_t petal_bkw_frw = 0, // = 1(backward) 2(forward) 0(all)
                           uint32_t petal         = 0,
                           uint32_t ring          = 0,
                           uint32_t ster          = 0) const; // ster = 1(mono) 2(stereo) 0(all)

   private:
      SiStripSubStructure(const SiStripSubStructure&); // stop default

      const SiStripSubStructure& operator=(const SiStripSubStructure&); // stop default

      // ---------- member data --------------------------------

};

#endif
