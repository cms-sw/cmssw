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
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/composite_key.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include "boost/tuple/tuple.hpp"

#include <boost/shared_ptr.hpp>

#include "DQM/CSCMonitorModule/interface/CSCDQM_Logger.h"
#include "DQM/CSCMonitorModule/interface/CSCDQM_HistoDef.h"
#include "DQM/CSCMonitorModule/interface/CSCDQM_MonitorObject.h"
#include "DQM/CSCMonitorModule/interface/CSCDQM_Utility.h"

namespace cscdqm {

  class CacheItem {
    public:
      virtual ~CacheItem() { }
      virtual void setMO(MonitorObject*) { }
      virtual void setMO(HwId, MonitorObject*) { }
      virtual void setMO(HwId, HwId, HwId, MonitorObject*) { }
      virtual const bool getMO(MonitorObject*&) const { return false; }
      virtual const bool getMO(HwId, MonitorObject*&) const { return false; }
      virtual const bool getMO(HwId, HwId, HwId, MonitorObject*&) const { return false; }
  };

  class EMUCacheItem : public CacheItem {

    private:

      MonitorObject* mo;

    public:

      EMUCacheItem(MonitorObject* mo_) : mo(mo_) { }
      void setMO(MonitorObject* mo_) { mo = mo_; }
      const bool getMO(MonitorObject*& mo_) const { 
        mo_ = mo; 
        return true;
      }

  };

  class ParCacheItem : public CacheItem {

    private:

      MonitorObject* mo;

    public:

      ParCacheItem(MonitorObject* mo_) : mo(mo_) { }
      void setMO(MonitorObject* mo_) { mo = mo_; }
      const bool getMO(MonitorObject*& mo_) const { 
        mo_ = mo; 
        return true;
      }

  };

  typedef std::map<HwId, MonitorObject*> DDUCacheMapType;

  class DDUCacheItem : public CacheItem {

    private:

      DDUCacheMapType mos;

    public:

      DDUCacheItem(HwId dduId, MonitorObject* mo) { 
        mos[dduId] = mo;
      }

      void setMO(HwId dduId, MonitorObject* mo) { 
        mos[dduId] = mo; 
      }

      const bool getMO(HwId dduId, MonitorObject*& mo_) const { 
        DDUCacheMapType::const_iterator it = mos.find(dduId);
        if (it != mos.end()) {
          mo_ = it->second;
          return true;
        }
        return false;
      }

  };

  typedef struct CSCKeyType {
    HwId crateId;
    HwId dmbId;
    HwId addId;
    MonitorObject* mo;
    CSCKeyType(HwId crateId_, HwId dmbId_, HwId addId_, MonitorObject* mo_) :
      crateId(crateId_), dmbId(dmbId_), addId(addId_), mo(mo_) { }
  };

  typedef boost::multi_index_container<
    CSCKeyType,
    boost::multi_index::indexed_by<
      boost::multi_index::ordered_unique< 
        boost::multi_index::composite_key<
          CSCKeyType,
          boost::multi_index::member<CSCKeyType, HwId, &CSCKeyType::crateId>,
          boost::multi_index::member<CSCKeyType, HwId, &CSCKeyType::dmbId>,
          boost::multi_index::member<CSCKeyType, HwId, &CSCKeyType::addId>
        >
      >
    >
  > CSCCacheMapType;

  class CSCCacheItem : public CacheItem {

    private:

      CSCCacheMapType mos;

    public:

      CSCCacheItem(HwId crateId, HwId dmbId, HwId addId, MonitorObject* mo) { 
        mos.insert(CSCKeyType(crateId, dmbId, addId, mo));
      }

      void setMO(HwId crateId, HwId dmbId, HwId addId, MonitorObject* mo) {
        mos.insert(CSCKeyType(crateId, dmbId, addId, mo));
      }

      const bool getMO(HwId crateId, HwId dmbId, HwId addId, MonitorObject*& mo_) const { 
        CSCCacheMapType::const_iterator it = mos.find(boost::make_tuple(crateId, dmbId, addId));
        if (it != mos.end()) {
          mo_ = it->mo;
          return true;
        }
        return false;
      }

  };
  

  /** DDU Key structure */
  typedef std::set<HwId> DDUSetType;
  typedef struct CSCIdType {
    HwId crateId;
    HwId dmbId;
    CSCIdType(HwId crateId_, HwId dmbId_) : crateId(crateId_), dmbId(dmbId_) { }
  };

  typedef boost::multi_index_container<
    CSCIdType,
    boost::multi_index::indexed_by<
      boost::multi_index::ordered_unique< 
        boost::multi_index::composite_key<
          CSCIdType,
          boost::multi_index::member<CSCIdType, HwId, &CSCIdType::crateId>,
          boost::multi_index::member<CSCIdType, HwId, &CSCIdType::dmbId>
        >
      >
    >
  > CSCSetType;

  /**
   * @class Cache
   * @brief MonitorObject cache - lists and routines to manage cache
   */
  class Cache {

    private:

      CacheItem* data[h::namesSize];

      DDUSetType ddus;
      CSCSetType cscs;

    public:
      
      Cache() {
        for (unsigned int i = 0; i < h::namesSize; i++) data[i] = 0;
      }

      ~Cache() {
        for (unsigned int i = 0; i < h::namesSize; i++) if (data[i]) delete data[i];
      }

      const bool get(const HistoDef& histo, MonitorObject*& mo);
      const bool getEMU(const HistoId& id, MonitorObject*& mo);
      const bool getDDU(const HistoId& id, const HwId& dduId, MonitorObject*& mo);
      const bool getCSC(const HistoId& id, const HwId& crateId, const HwId& dmbId, const HwId& addId, MonitorObject*& mo);
      const bool getPar(const HistoId& id, MonitorObject*& mo);
      void put(const HistoDef& histo, MonitorObject* mo);

      const bool nextBookedDDU(unsigned int& n, unsigned int& dduId) const;
      const bool nextBookedCSC(unsigned int& n, unsigned int& crateId, unsigned int& dmbId) const;
      const bool isBookedCSC(const HwId& crateId, const HwId& dmbId) const;
      const bool isBookedDDU(const HwId& dduId) const;

  };

}

#endif
