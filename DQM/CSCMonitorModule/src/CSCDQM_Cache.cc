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

  const bool Cache::get(const HistoType& histo, MonitorObject*& mo) {
    CacheMap::iterator it = cache.find(histo.getFullPath());
    if (it == cache.end()) return false;
    mo = it->second;
    return true; 
  }

  void Cache::put(const HistoType& histo, MonitorObject* mo) {
    cache[histo.getFullPath()] = mo;
  }

}
