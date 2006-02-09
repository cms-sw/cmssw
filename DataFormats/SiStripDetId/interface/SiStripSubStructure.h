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
// $Id$
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
                           uint32_t string        = 0,
                           uint32_t mod_in_string = 0,
                           uint32_t ster          = 0) const;

      void getTIDDetectors(const std::vector<uint32_t> & inputDetRawIds, // input
                           std::vector<uint32_t> & tidDetRawIds,         // output
                           uint32_t side        = 0,                     // selection
                           uint32_t wheel       = 0,
                           uint32_t ring        = 0,
                           uint32_t mod_in_ring = 0,
                           uint32_t ster        = 0) const;

      void getTOBDetectors(const std::vector<uint32_t> & inputDetRawIds, // input
                           std::vector<uint32_t> & tobDetRawIds,         // output
                           uint32_t layer      = 0,                      // selection
                           uint32_t rod        = 0,
                           uint32_t mod_in_rod = 0,
                           uint32_t ster       = 0) const;

      void getTECDetectors(const std::vector<uint32_t> & inputDetRawIds, // input
                           std::vector<uint32_t> & tecDetRawIds,         // output
                           uint32_t side        = 0,                     // selection
                           uint32_t wheel       = 0,
                           uint32_t petal       = 0,
                           uint32_t ring        = 0,
                           uint32_t mod_in_ring = 0,
                           uint32_t ster        = 0) const;

   private:
      SiStripSubStructure(const SiStripSubStructure&); // stop default

      const SiStripSubStructure& operator=(const SiStripSubStructure&); // stop default

      // ---------- member data --------------------------------

};

#endif
