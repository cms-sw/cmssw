#ifndef SiStripCommon_SiStripHistoIdManager_h
#define SiStripCommon_SiStripHistoIdManager_h
// -*- C++ -*-
//
// Package:     SiStripCommon
// Class  :     SiStripHistoIdManager
// 
/**\class SiStripHistoIdManager SiStripHistoIdManager.h DQM/SiStripCommon/interface/SiStripHistoIdManager.h

 Description: <Create a histogram id from a descriptive string and an integer id of the component (detector, fed, fec, etc.). Also extract integer id of the component from such a title.>

 Usage:
    <usage>

*/
//
// Original Author:  dkcira
//         Created:  Sat Feb  4 15:42:46 CET 2006
// $Id$
//

#include <string>

class SiStripHistoIdManager
{
   public:
      SiStripHistoIdManager();
      virtual ~SiStripHistoIdManager();
      // create a histogram id from a descriptive string and an integer id of the component (detector, fed, fec, etc.)
      // generally: histoid = description + separator + component_id
      std::string createHistoId(std::string description, uint32_t component_id);
      // extract the component id from a histogram id
      uint32_t    getComponentId(std::string histoid);
   private:
      SiStripHistoIdManager(const SiStripHistoIdManager&); // stop default
      const SiStripHistoIdManager& operator=(const SiStripHistoIdManager&); // stop default
      // ---------- member data --------------------------------
      std::string separator;
};


#endif
