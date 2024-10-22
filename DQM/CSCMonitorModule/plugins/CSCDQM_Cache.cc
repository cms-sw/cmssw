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

#include "CSCDQM_Cache.h"

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
    } else if (typeid(histo) == FEDHistoDefT) {
      return getFED(histo.getId(), histo.getFEDId(), mo);
    } else if (typeid(histo) == DDUHistoDefT) {
      return getDDU(histo.getId(), histo.getDDUId(), mo);
    } else if (typeid(histo) == CSCHistoDefT) {
      return getCSC(histo.getId(), histo.getCrateId(), histo.getDMBId(), histo.getAddId(), mo);
    } else if (typeid(histo) == ParHistoDefT) {
      return getPar(histo.getId(), mo);
    }

    return false;
  }

  /**
   * @brief  Get EMU MO on Histogram Id
   * @param  id Histogram identifier
   * @param  mo Monitoring Object to return
   * @return true if MO was found in cache and false otherwise
   */
  const bool Cache::getEMU(const HistoId& id, MonitorObject*& mo) {
    if (data[id]) {
      mo = data[id];
      return true;
    }
    return false;
  }

  /**
   * @brief  Get FED MO on Histogram Id and FED Id
   * @param  id Histogram identifier
   * @param  fedId FED identifier
   * @param  mo Monitoring Object to return
   * @return true if MO was found in cache and false otherwise
   */
  const bool Cache::getFED(const HistoId& id, const HwId& fedId, MonitorObject*& mo) {
    /** If not cached (last FED) - find FED */
    if (fedPointerValue != fedId) {
      fedPointer = fedData.find(fedId);
      if (fedPointer == fedData.end()) {
        fedPointerValue = 0;
        return false;
      }
      fedPointerValue = fedId;
    }

    /** Get MO from static array */
    if (fedPointer->second[id]) {
      mo = fedPointer->second[id];
      return true;
    }
    return false;
  }

  /**
   * @brief  Get DDU MO on Histogram Id and DDU Id
   * @param  id Histogram identifier
   * @param  dduId DDU identifier
   * @param  mo Monitoring Object to return
   * @return true if MO was found in cache and false otherwise
   */
  const bool Cache::getDDU(const HistoId& id, const HwId& dduId, MonitorObject*& mo) {
    /** If not cached (last DDU) - find DDU */
    if (dduPointerValue != dduId) {
      dduPointer = dduData.find(dduId);
      if (dduPointer == dduData.end()) {
        dduPointerValue = 0;
        return false;
      }
      dduPointerValue = dduId;
    }

    /** Get MO from static array */
    if (dduPointer->second[id]) {
      mo = dduPointer->second[id];
      return true;
    }
    return false;
  }

  /**
   * @brief  Get CSC MO on Histogram Id and CSC Crate and DMB Ids
   * @param  id Histogram identifier
   * @param  crateId CSC Crate identifier
   * @param  dmbId CSC DMB identifier
   * @param  mo Monitoring Object to return
   * @return true if MO was found in cache and false otherwise
   */
  const bool Cache::getCSC(
      const HistoId& id, const HwId& crateId, const HwId& dmbId, const HwId& addId, MonitorObject*& mo) {
    /** If not cached (last CSC) - find CSC */
    if (cscPointer == cscData.end() || cscPointer->crateId != crateId || cscPointer->dmbId != dmbId) {
      cscPointer = cscData.find(boost::make_tuple(crateId, dmbId));
    }

    /** Get Monitor object from multi_index List */
    if (cscPointer != cscData.end()) {
      CSCHistoMapType::const_iterator hit = cscPointer->mos.find(boost::make_tuple(id, addId));
      if (hit != cscPointer->mos.end()) {
        mo = const_cast<MonitorObject*>(hit->mo);
        return true;
      }
    }
    return false;
  }

  /**
   * @brief  Get Parameter MO on Histogram Id
   * @param  id Histogram identifier
   * @param  mo Monitoring Object to return
   * @return true if MO was found in cache and false otherwise
   */
  const bool Cache::getPar(const HistoId& id, MonitorObject*& mo) {
    if (data[id]) {
      mo = data[id];
      return true;
    }
    return false;
  }

  /**
   * @brief  Put Monitoring Object into cache
   * @param  histo Histogram Definition
   * @param  mo Monitoring Object to put
   * @return
   */
  void Cache::put(const HistoDef& histo, MonitorObject* mo) {
    HistoId id = histo.getId();

    /** EMU MO */
    if (typeid(histo) == EMUHistoDefT) {
      data[id] = mo;
    } else

      /** FED MO */
      if (typeid(histo) == FEDHistoDefT) {
        HwId fedId = histo.getFEDId();

        if (fedPointerValue != fedId) {
          fedPointer = fedData.find(fedId);
        }

        if (fedPointer == fedData.end()) {
          MonitorObject** mos = new MonitorObject*[h::namesSize];
          for (unsigned int i = 0; i < h::namesSize; i++)
            mos[i] = nullptr;
          fedPointer = fedData.insert(fedData.end(), std::make_pair(fedId, mos));
        }

        fedPointer->second[id] = mo;
        fedPointerValue = fedId;

      } else

        /** DDU MO */
        if (typeid(histo) == DDUHistoDefT) {
          HwId dduId = histo.getDDUId();

          if (dduPointerValue != dduId) {
            dduPointer = dduData.find(dduId);
          }

          if (dduPointer == dduData.end()) {
            MonitorObject** mos = new MonitorObject*[h::namesSize];
            for (unsigned int i = 0; i < h::namesSize; i++)
              mos[i] = nullptr;
            dduPointer = dduData.insert(dduData.end(), std::make_pair(dduId, mos));
          }

          dduPointer->second[id] = mo;
          dduPointerValue = dduId;

        } else

          /** CSC MO */
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

            /** Parameter MO */
            if (typeid(histo) == ParHistoDefT) {
              data[id] = mo;
            }

    /** Add histo (if mo is not null!) into lookup list */
    if (mo) {
      lookupData.insert(lookupData.end(), LookupKeyType(histo, mo));
    }
  }

  /**
   * @brief  Iterator to get booked CSC identifiers on enumerator
   * @param  n iterator (0 and up)
   * @param  crateId CSC Crate Id returned
   * @param  dmbId CSC DMB Id returned
   * @return true if CSC on n found, false - otherwise
   */
  const bool Cache::nextBookedCSC(unsigned int& n, unsigned int& crateId, unsigned int& dmbId) const {
    if (n < cscData.size()) {
      CSCMapType::const_iterator iter = cscData.begin();
      for (unsigned int i = n; i > 0; i--)
        iter++;
      crateId = iter->crateId;
      dmbId = iter->dmbId;
      n++;
      return true;
    }
    return false;
  }

  /**
   * @brief  Iterator to get booked FED identifier on enumerator
   * @param  n iterator (0 and up)
   * @param  fedId FED Id returned
   * @return true if FED on n found, false - otherwise
   */
  const bool Cache::nextBookedFED(unsigned int& n, unsigned int& fedId) const {
    if (n < fedData.size()) {
      FEDMapType::const_iterator iter = fedData.begin();
      for (unsigned int i = n; i > 0; i--)
        iter++;
      fedId = iter->first;
      n++;
      return true;
    }
    return false;
  }

  /**
   * @brief  Iterator to get booked DDU identifier on enumerator
   * @param  n iterator (0 and up)
   * @param  dduId DDU Id returned
   * @return true if DDU on n found, false - otherwise
   */
  const bool Cache::nextBookedDDU(unsigned int& n, unsigned int& dduId) const {
    if (n < dduData.size()) {
      DDUMapType::const_iterator iter = dduData.begin();
      for (unsigned int i = n; i > 0; i--)
        iter++;
      dduId = iter->first;
      n++;
      return true;
    }
    return false;
  }

  /**
   * @brief  Check if CSC was booked on given identifiers 
   * @param  crateId CSC Crate Id
   * @param  dmbId CSC DMB Id
   * @return true if CSC was booked, false - otherwise
   */
  const bool Cache::isBookedCSC(const HwId& crateId, const HwId& dmbId) const {
    CSCMapType::const_iterator it = cscData.find(boost::make_tuple(crateId, dmbId));
    if (it != cscData.end()) {
      return true;
    }
    return false;
  }

  /**
   * @brief  Check if FED was booked on given identifier 
   * @param  fedId FED Id
   * @return true if FED was booked, false - otherwise
   */
  const bool Cache::isBookedFED(const HwId& fedId) const {
    FEDMapType::const_iterator iter = fedData.find(fedId);
    return (iter != fedData.end());
  }

  /**
   * @brief  Check if DDU was booked on given identifier 
   * @param  dduId DDU Id
   * @return true if DDU was booked, false - otherwise
   */
  const bool Cache::isBookedDDU(const HwId& dduId) const {
    DDUMapType::const_iterator iter = dduData.find(dduId);
    return (iter != dduData.end());
  }

}  // namespace cscdqm
