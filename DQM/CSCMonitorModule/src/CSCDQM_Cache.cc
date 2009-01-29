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

  const bool Cache::getEMU(const HistoId& id, MonitorObject*& mo) {
    if (data[id]) {
      mo = data[id];
      return true;
    }
    return false;
  }

  const bool Cache::getDDU(const HistoId& id, const HwId& dduId, MonitorObject*& mo) {

    if (dduPointerValue != dduId) {
      dduPointer = dduData.find(dduId);
      if (dduPointer == dduData.end()) {
        dduPointerValue = 0;
        return false;
      }
      dduPointerValue  = dduId;
    } 

    if (dduPointer->second[id]) {
      mo = dduPointer->second[id];
      return true;
    }
    return false;

  }

  const bool Cache::getCSC(const HistoId& id, const HwId& crateId, const HwId& dmbId, const HwId& addId, MonitorObject*& mo) {

    if (cscPointer == cscData.end() || cscPointer->crateId != crateId || cscPointer->dmbId != dmbId) {
      cscPointer = cscData.find(boost::make_tuple(crateId, dmbId));
    }

    if (cscPointer != cscData.end()) {
      CSCHistoMapType::const_iterator hit = cscPointer->mos.find(boost::make_tuple(id, addId));
      if (hit != cscPointer->mos.end()) {
        mo = const_cast<MonitorObject*>(hit->mo);
        return true;
      }
    }
    return false;
  }

  const bool Cache::getPar(const HistoId& id, MonitorObject*& mo) {
    if (data[id]) {
      mo = data[id];
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

    HistoId id = histo.getId();

    if (typeid(histo) == EMUHistoDefT) {
      data[id] = mo;
    } else

    if (typeid(histo) == DDUHistoDefT) {

      HwId dduId = histo.getDDUId();

      if (dduPointerValue != dduId) {
        dduPointer = dduData.find(dduId);
      } 

      if (dduPointer == dduData.end()) {
        MonitorObject** mos = new MonitorObject*[h::namesSize];
        for (unsigned int i = 0; i < h::namesSize; i++) mos[i] = 0;
        dduPointer = dduData.insert(dduData.end(), std::make_pair(dduId, mos));
      }

      dduPointer->second[id] = mo;
      dduPointerValue = dduId;

    } else

    if (typeid(histo) == CSCHistoDefT) {

      HwId crateId = histo.getCrateId();
      HwId dmbId = histo.getDMBId();
      HwId addId = histo.getAddId();

      CSCHistoKeyType histoKey(id, addId, mo);

      if (cscPointer == cscData.end() || cscPointer->crateId != crateId || cscPointer->dmbId != dmbId) {
        cscPointer = cscData.find(boost::make_tuple(crateId, dmbId));
      } 

      if (cscPointer == cscData.end()) {
        CSCKeyType cscKey(crateId, dmbId);
        cscPointer = cscData.insert(cscData.end(), cscKey);
      }
      CSCHistoMapType* mos = const_cast<CSCHistoMapType*>(&cscPointer->mos);
      mos->insert(histoKey);

    } else

    if (typeid(histo) == ParHistoDefT) {
      data[id] = mo;
    }

  }

  const bool Cache::nextBookedCSC(unsigned int& n, unsigned int& crateId, unsigned int& dmbId) const {
    if (n < cscData.size()) {
      CSCMapType::const_iterator iter = cscData.begin();
      for (unsigned int i = n; i > 0; i--) iter++;
      crateId = iter->crateId;
      dmbId   = iter->dmbId;
      n++;
      return true;
    }
    return false;
  }

  const bool Cache::nextBookedDDU(unsigned int& n, unsigned int& dduId) const {
    if (n < dduData.size()) {
      DDUMapType::const_iterator iter = dduData.begin();
      for (unsigned int i = n; i > 0; i--) iter++;
      dduId = iter->first;
      n++;
      return true;
    }
    return false;
  }

  const bool Cache::isBookedCSC(const HwId& crateId, const HwId& dmbId) const {
    CSCMapType::const_iterator it = cscData.find(boost::make_tuple(crateId, dmbId));
    if (it != cscData.end()) {
      return true;
    }
    return false;
  }

  const bool Cache::isBookedDDU(const HwId& dduId) const {
    DDUMapType::const_iterator iter = dduData.find(dduId);
    return (iter != dduData.end());
  }

}
