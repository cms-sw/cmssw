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

  /**
   * @brief  Get Monitoring Object on Histogram Definition
   * @param  histo Histogram definition
   * @param  mo Monitoring Object to return
   * @return true if MO was found in cache and false otherwise
   */
  const bool Cache::get(const HistoDef& histo, MonitorObject*& mo) {
    HistoCacheKey key(histo);
    bool found = get(key, mo);
    /*
    if (found) {
        LOG_DEBUG << "CACHE: histo " << histo << " key " << key << " found in cache: " << mo;
    } else {
        LOG_DEBUG << "CACHE: histo " << histo << " key " << key << " NOT found in cache.";
    }
    */
    return found;
  }

  /**
   * @brief  Get Monitoring Object on separate MO identification elements
   * @param  id Histogram identification
   * @param  mo Monitoring Object to return
   * @param  id1 first identifier (DDU id)
   * @param  id2 second identifier (Chamber id)
   * @param  id3 third identifier (DMB id)
   * @param  id4 fourth identifier (Additional id)
   * @return true if MO was found in cache and false otherwise
   */
  const bool Cache::get(const HistoId id, MonitorObject*& mo, const HwId& id1, const HwId& id2, const HwId& id3, const HwId& id4) {
    return get(HistoCacheKey(id, id1, id2, id3, id4), mo);
  }

  /**
   * @brief  Get Monitoring Object on Monitoring Object key
   * @param  key Monitoring Object key
   * @param  mo Monitoring Object to return
   * @return true if MO was found in cache and false otherwise
   */
  const bool Cache::get(const HistoCacheKey& key, MonitorObject*& mo) {
    CacheMap::iterator it = cache.find(boost::make_tuple(key.id, key.id1, key.id2, key.id3, key.id4));
    if (it != cache.end()) {
      mo = it->mop.get();
      return true;
    }
    return false;
  }

  /**
   * @brief  Put Monitoring Object into cache
   * @param  histo Histogram Definition (to be used to generate cache key)
   * @param  mo Monitoring Object to put
   * @return
   */
  void Cache::put(const HistoDef& histo, MonitorObject* mo) {
    HistoCacheKey key(histo, mo);
    cache.insert(key);
    //LOG_DEBUG << "CACHE: histo " << histo << " key " << key << " was put to cache (" << mo << ")";
  }

  /**
   * @brief  Print Cache content (used for debugging purposes only)
   * @param  
   * @return 
   */
  void Cache::printContent() const {
    for (CacheMap::const_iterator it = cache.begin(); it != cache.end(); it++) {
      LOG_DEBUG << "CACHE: content = " << *it;
    }
  }

}
