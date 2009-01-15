/*
 * =====================================================================================
 *
 *       Filename:  CSCDQM_Cache.cc
 *
 *    Description:  MonitorObject cache implementation
 *
 *        Version:  1.0
 *        Created:  12/01/2008 11:36:11 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius (VR), valdas.rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */

#include "DQM/CSCMonitorModule/interface/CSCDQM_Cache.h"

namespace cscdqm {

  const bool Cache::get(const HistoDef& histo, MonitorObject*& mo) {
    CacheMap::iterator it = cache.find(HistoCacheKey(histo));
    if (it == cache.end()) return false;
    mo = it->second.get();
    return true; 
  }

  const bool Cache::get(const HistoId id, MonitorObject*& mo, const HwId id1, const HwId id2, const HwId id3, const HwId id4) {
    CacheMap::iterator it = cache.find(HistoCacheKey(id, id1, id2, id3, id4));
    if (it == cache.end()) return false;
    mo = it->second.get();
    return true; 
  }

  void Cache::put(const HistoDef& histo, MonitorObject* mo) {
    cache[HistoCacheKey(histo)] = MonitorObjectPtr(mo);
  }

}
