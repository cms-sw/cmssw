/*
 * =====================================================================================
 *
 *       Filename:  CSCDQM_Cache.h
 *
 *    Description:  Efficiently manages lists of MonitorObject's for internal
 *    MO cache.
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

#include "CSCDQM_Logger.h"
#include "CSCDQM_HistoDef.h"
#include "CSCDQM_MonitorObject.h"
#include "CSCDQM_Utility.h"

namespace cscdqm {

  /** @brief Chamber MO List object definition */
  struct CSCHistoKeyType {
    HistoId id;
    HwId addId;
    const MonitorObject* mo;
    CSCHistoKeyType(const HistoId& id_, const HwId& addId_, const MonitorObject* mo_)
        : id(id_), addId(addId_), mo(mo_) {}
  };

  /** Chamber MO List definition */
  typedef boost::multi_index_container<
      CSCHistoKeyType,
      boost::multi_index::indexed_by<boost::multi_index::ordered_unique<boost::multi_index::composite_key<
          CSCHistoKeyType,
          boost::multi_index::member<CSCHistoKeyType, HistoId, &CSCHistoKeyType::id>,
          boost::multi_index::member<CSCHistoKeyType, HwId, &CSCHistoKeyType::addId> > > > >
      CSCHistoMapType;

  /** @brief Chamber List object definition */
  struct CSCKeyType {
    HwId crateId;
    HwId dmbId;
    CSCHistoMapType mos;
    CSCKeyType(const HwId& crateId_, const HwId& dmbId_) : crateId(crateId_), dmbId(dmbId_) {}
  };

  /** Chamber List definition */
  typedef boost::multi_index_container<
      CSCKeyType,
      boost::multi_index::indexed_by<boost::multi_index::ordered_unique<
          boost::multi_index::composite_key<CSCKeyType,
                                            boost::multi_index::member<CSCKeyType, HwId, &CSCKeyType::crateId>,
                                            boost::multi_index::member<CSCKeyType, HwId, &CSCKeyType::dmbId> > > > >
      CSCMapType;

  /** FED List definition (static MO list) */
  typedef std::map<HwId, MonitorObject**> FEDMapType;

  /** DDU List definition (static MO list) */
  typedef std::map<HwId, MonitorObject**> DDUMapType;

  /** @brief MO Lookup List object definition */
  struct LookupKeyType {
    HistoDef histo;
    std::string path;
    MonitorObject* mo;
    LookupKeyType(const HistoDef& histo_, MonitorObject*& mo_) : histo(histo_), mo(mo_) { path = histo.getPath(); }
  };

  /** MO Lookup List definition */
  typedef boost::multi_index_container<
      LookupKeyType,
      boost::multi_index::indexed_by<
          boost::multi_index::ordered_unique<boost::multi_index::member<LookupKeyType, HistoDef, &LookupKeyType::histo> >,
          boost::multi_index::ordered_non_unique<
              boost::multi_index::member<LookupKeyType, std::string, &LookupKeyType::path> > > >
      LookupMapType;

  /**
   * @class Cache
   * @brief MonitorObject cache - list objects and routines to manage cache
   */
  class Cache {
  private:
    /** EMU and PAR MO static List */
    MonitorObject* data[h::namesSize];

    /** FED MO List */
    FEDMapType fedData;
    /** Pointer to the Last FED object used (cached) */
    FEDMapType::const_iterator fedPointer;
    /** Last FED id used (cached) */
    HwId fedPointerValue;

    /** DDU MO List */
    DDUMapType dduData;
    /** Pointer to the Last DDU object used (cached) */
    DDUMapType::const_iterator dduPointer;
    /** Last DDU id used (cached) */
    HwId dduPointerValue;

    /** Chamber MO List */
    CSCMapType cscData;
    /** Pointer to the Last Chamber object used (cached) */
    CSCMapType::const_iterator cscPointer;

    /** MO Lookup List */
    LookupMapType lookupData;

  public:
    /** Cache Constructor */
    Cache() {
      /** Initialize EMU and PAR static array with zero's */
      for (unsigned int i = 0; i < h::namesSize; i++)
        data[i] = nullptr;

      /** Initialize FED cached pointers */
      fedPointer = fedData.end();
      fedPointerValue = 0;

      /** Initialize DDU and CSC cached pointers */
      dduPointer = dduData.end();
      dduPointerValue = 0;
      cscPointer = cscData.end();
    }

    /** Destructor */
    ~Cache() {
      /** Clear FED MO static arrays */
      while (fedData.begin() != fedData.end()) {
        if (fedData.begin()->second) {
          delete[] fedData.begin()->second;
        }
        fedData.erase(fedData.begin());
      }

      /** Clear DDU MO static arrays */
      while (dduData.begin() != dduData.end()) {
        if (dduData.begin()->second) {
          delete[] dduData.begin()->second;
        }
        dduData.erase(dduData.begin());
      }
    }

    /** Native Cache methods */

    const bool get(const HistoDef& histo, MonitorObject*& mo);
    const bool getEMU(const HistoId& id, MonitorObject*& mo);
    const bool getFED(const HistoId& id, const HwId& fedId, MonitorObject*& mo);
    const bool getDDU(const HistoId& id, const HwId& dduId, MonitorObject*& mo);
    const bool getCSC(const HistoId& id, const HwId& crateId, const HwId& dmbId, const HwId& addId, MonitorObject*& mo);
    const bool getPar(const HistoId& id, MonitorObject*& mo);
    void put(const HistoDef& histo, MonitorObject* mo);

    /** Utility methods */

    const bool nextBookedFED(unsigned int& n, unsigned int& fedId) const;
    const bool nextBookedDDU(unsigned int& n, unsigned int& dduId) const;
    const bool nextBookedCSC(unsigned int& n, unsigned int& crateId, unsigned int& dmbId) const;
    const bool isBookedCSC(const HwId& crateId, const HwId& dmbId) const;
    const bool isBookedDDU(const HwId& dduId) const;
    const bool isBookedFED(const HwId& fedId) const;
  };

}  // namespace cscdqm

#endif
