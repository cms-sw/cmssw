#ifndef SiStripCommon_SiStripHistoId_h
#define SiStripCommon_SiStripHistoId_h
// -*- C++ -*-
//
// Package:     SiStripCommon
// Class  :     SiStripHistoId
// 
/**\class SiStripHistoId SiStripHistoId.h DQM/SiStripCommon/interface/SiStripHistoId.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  dkcira
//         Created:  Wed Feb 22 16:07:51 CET 2006
//

#include <string>
#include <boost/cstdint.hpp>


class TrackerTopology;
class SiStripHistoId
{
   public:
      SiStripHistoId();
      virtual ~SiStripHistoId();
      // generally: histoid = description + separator1 + id_type + separator2 + component_id
      std::string createHistoId(std::string description, std::string id_type, uint32_t component_id);
      std::string createHistoLayer(std::string description, std::string id_type,std::string path, std::string flag);
      //      std::string getSubdetid(uint32_t id, const TrackerTopology* tTopo, bool flag_ring, bool flag_thickness = false);
      std::string getSubdetid(uint32_t id, const TrackerTopology* tTopo, bool flag_ring);
      // extract the component_id and the id_type from a histogram id
      uint32_t    getComponentId(std::string histoid);
      std::string getComponentType(std::string histoid);
   private:
      SiStripHistoId(const SiStripHistoId&); // stop default
      const SiStripHistoId& operator=(const SiStripHistoId&); // stop default
      std::string returnIdPart(std::string histoid, uint32_t whichpart);
};

#endif
