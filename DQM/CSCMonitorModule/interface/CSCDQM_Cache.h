/*
 * =====================================================================================
 *
 *       Filename:  CSCDQM_Cache.h
 *
 *    Description:  Efficiently manages lists of MonitorObject 's 
 *
 *        Version:  1.0
 *        Created:  11/27/2008 10:05:00 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius (VR), valdas.rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */

#ifndef CSCDQM_Cache_H
#define CSCDQM_Cache_H 

#include <map>
#include <set>

#include <boost/shared_ptr.hpp>

#include "DQM/CSCMonitorModule/interface/CSCDQM_Logger.h"
#include "DQM/CSCMonitorModule/interface/CSCDQM_HistoDef.h"
#include "DQM/CSCMonitorModule/interface/CSCDQM_MonitorObject.h"
#include "DQM/CSCMonitorModule/interface/CSCDQM_Utility.h"

namespace cscdqm {

  typedef struct HistoCacheKey {

    HistoId id;
    HwId    id1;
    HwId    id2;
    HwId    id3;
    HwId    id4;

    HistoCacheKey(const HistoDef& h) {
      id  = h.getId();
      id1 = h.getDDUId();
      id2 = h.getCrateId();
      id3 = h.getDMBId();
      id4 = h.getAddId();
    }

    HistoCacheKey(const HistoId p_id, const HwId p_id1, const HwId p_id2, const HwId p_id3, const HwId p_id4) {
      id  = p_id;
      id1 = p_id1;
      id2 = p_id2;
      id3 = p_id3;
      id4 = p_id4;
    }

    const HistoCacheKey& operator= (const HistoCacheKey& k) {
      id = k.id; 
      id1 = k.id1; 
      id2 = k.id2; 
      id3 = k.id3; 
      id4 = k.id4;
      return *this;
    }

    const bool operator== (const HistoCacheKey& k) const {
      return (id == k.id && id1 == k.id1 && id2 == k.id2 && id3 == k.id3 && id4 == k.id4);
    }

    const bool operator< (const HistoCacheKey& k) const {
      if (id < k.id)   return true;
      if (id1 < k.id1) return true;
      if (id2 < k.id2) return true; 
      if (id3 < k.id3) return true;
      if (id4 < k.id4) return true;
      return false;
    }

  };

  typedef boost::shared_ptr<MonitorObject> MonitorObjectPtr;
  typedef std::map<const HistoCacheKey, MonitorObjectPtr> CacheMap;

  /**
   * @class Cache
   * @brief MonitorObject cache - lists and routines to manage cache
   */
  class Cache {

    private:

      CacheMap cache;

    public:
      
      const bool get(const HistoDef& histo, MonitorObject*& mo);
      const bool get(const HistoId id, MonitorObject*& mo, const HwId id1 = 0, const HwId id2 = 0, const HwId id3 = 0, const HwId id4 = 0);
      void put(const HistoDef& histo, MonitorObject* mo);

  };

}

#endif
