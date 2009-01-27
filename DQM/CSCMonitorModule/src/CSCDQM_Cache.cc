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

    if (typeid(histo) == EMUHistoDefT) {
      return getEMU(histo.getId(), mo);
    } else
    if (typeid(histo) == DDUHistoDefT) {
      return getDDU(histo.getId(), histo.getDDUId(), mo);
    } else
    if (typeid(histo) == CSCHistoDefT) {
      return getCSC(histo.getId(), histo.getCrateId(), histo.getDMBId(), histo.getAddId(), mo);
    } else
    if (typeid(histo) == ParHistoDefT) {
      return getPar(histo.getId(), mo);
    }

    /*
    if (found) {
        LOG_DEBUG << "CACHE: histo " << histo << " key " << key << " found in cache: " << mo;
    } else {
        LOG_DEBUG << "CACHE: histo " << histo << " key " << key << " NOT found in cache.";
    }
    */
    
    return false;
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
  const bool Cache::getEMU(const HistoId& id, MonitorObject*& mo) {
    if (data[id]) {
      return data[id]->getMO(mo);
    }
    return false;
  }

  const bool Cache::getDDU(const HistoId& id, const HwId& dduId, MonitorObject*& mo) {
    if (data[id]) {
      return data[id]->getMO(dduId, mo);
    }
    return false;
  }

  const bool Cache::getCSC(const HistoId& id, const HwId& crateId, const HwId& dmbId, const HwId& addId, MonitorObject*& mo) {
    if (data[id]) {
      return data[id]->getMO(crateId, dmbId, addId, mo);
    }
    return false;
  }

  const bool Cache::getPar(const HistoId& id, MonitorObject*& mo) {
    if (data[id]) {
      return data[id]->getMO(mo);
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

    HistoId id = histo.getId();

    if (typeid(histo) == EMUHistoDefT) {
      if (data[id]) {
        data[id]->setMO(mo);
      } else {
        data[id] = new EMUCacheItem(mo);
      }
    } else

    if (typeid(histo) == DDUHistoDefT) {
      HwId dduId = histo.getDDUId();
      if (data[id]) {
        data[id]->setMO(dduId, mo);
      } else {
        data[id] = new DDUCacheItem(dduId, mo);
      }
      ddus.insert(dduId);
    } else

    if (typeid(histo) == CSCHistoDefT) {
      HwId crateId = histo.getCrateId();
      HwId dmbId = histo.getDMBId();
      HwId addId = histo.getAddId();
      if (data[id]) {
        data[id]->setMO(crateId, dmbId, addId, mo);
      } else {
        data[id] = new CSCCacheItem(crateId, dmbId, addId, mo);
      }
      cscs.insert(CSCIdType(crateId, dmbId));
    } else

    if (typeid(histo) == ParHistoDefT) {
      if (data[id]) {
        data[id]->setMO(mo);
      } else {
        data[id] = new ParCacheItem(mo);
      }
    }

  }

  const bool Cache::nextBookedCSC(unsigned int& n, unsigned int& crateId, unsigned int& dmbId) const {
    if (n < cscs.size()) {
      CSCSetType::const_iterator iter = cscs.begin();
      for (unsigned int i = n; i > 0; i--) iter++;
      crateId = iter->crateId;
      dmbId   = iter->dmbId;
      n++;
      return true;
    }
    return false;
  }

  const bool Cache::nextBookedDDU(unsigned int& n, unsigned int& dduId) const {
    if (n < ddus.size()) {
      DDUSetType::const_iterator iter = ddus.begin();
      for (unsigned int i = n; i > 0; i--) iter++;
      dduId = *iter;
      n++;
      return true;
    }
    return false;
  }

  const bool Cache::isBookedCSC(const HwId& crateId, const HwId& dmbId) const {
    CSCSetType::const_iterator it = cscs.find(boost::make_tuple(crateId, dmbId));
    if (it != cscs.end()) {
      return true;
    }
    return false;
  }

  const bool Cache::isBookedDDU(const HwId& dduId) const {
    DDUSetType::const_iterator iter = ddus.find(dduId);
    return (iter != ddus.end());
  }

}
