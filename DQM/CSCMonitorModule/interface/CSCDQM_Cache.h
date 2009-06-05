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

  typedef struct CSCHistoKeyType {
    HistoId id;
    HwId addId;
    const MonitorObject* mo;
    CSCHistoKeyType(const HistoId& id_, const HwId& addId_, const MonitorObject* mo_) : id(id_), addId(addId_), mo(mo_) { }
  };

  typedef boost::multi_index_container<
    CSCHistoKeyType,
    boost::multi_index::indexed_by<
      boost::multi_index::ordered_unique< 
        boost::multi_index::composite_key<
          CSCHistoKeyType,
          boost::multi_index::member<CSCHistoKeyType, HistoId, &CSCHistoKeyType::id>,
          boost::multi_index::member<CSCHistoKeyType, HwId, &CSCHistoKeyType::addId>
        >
      >
    >
  > CSCHistoMapType;

  typedef struct CSCKeyType {
    HwId crateId;
    HwId dmbId;
    CSCHistoMapType mos;
    CSCKeyType(const HwId& crateId_, const HwId& dmbId_) : crateId(crateId_), dmbId(dmbId_) { }
  };

  typedef boost::multi_index_container<
    CSCKeyType,
    boost::multi_index::indexed_by<
      boost::multi_index::ordered_unique< 
        boost::multi_index::composite_key<
          CSCKeyType,
          boost::multi_index::member<CSCKeyType, HwId, &CSCKeyType::crateId>,
          boost::multi_index::member<CSCKeyType, HwId, &CSCKeyType::dmbId>
        >
      >
    >
  > CSCMapType;
  
  typedef std::map<HwId, MonitorObject**> DDUMapType;

  /**
   * @class Cache
   * @brief MonitorObject cache - lists and routines to manage cache
   */
  class Cache {

    private:

      MonitorObject* data[h::namesSize];

      DDUMapType dduData;
      DDUMapType::const_iterator dduPointer;
      HwId dduPointerValue;

      CSCMapType cscData;
      CSCMapType::const_iterator cscPointer;

    public:
      
      Cache() {
        for (unsigned int i = 0; i < h::namesSize; i++) data[i] = 0;
        dduPointer = dduData.end();
        dduPointerValue = 0;
        cscPointer = cscData.end();
      }

      ~Cache() {
        DDUMapType::iterator it;
        while (dduData.size() > 0) {
          it = dduData.begin();
          if (it->second) delete [] it->second;
          dduData.erase(it);
        }
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
