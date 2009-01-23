/*
 * =====================================================================================
 *
 *       Filename:  CSCDQM_Dispatcher.h
 *
 *    Description:  Framework frontend and Histogram Cache controller
 *
 *        Version:  1.0
 *        Created:  10/03/2008 10:26:04 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius, valdas.rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */

#ifndef CSCDQM_Dispatcher_H
#define CSCDQM_Dispatcher_H

#include <typeinfo>

#include <boost/thread.hpp>
#include <boost/thread/recursive_mutex.hpp>
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/composite_key.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include "boost/tuple/tuple.hpp"

#include "DQM/CSCMonitorModule/interface/CSCDQM_Configuration.h"
#include "DQM/CSCMonitorModule/interface/CSCDQM_EventProcessor.h"
#include "DQM/CSCMonitorModule/interface/CSCDQM_Collection.h"
#include "DQM/CSCMonitorModule/interface/CSCDQM_Cache.h"
#include "DQM/CSCMonitorModule/interface/CSCDQM_Logger.h"
#include "DQM/CSCMonitorModule/interface/CSCDQM_Lock.h"

namespace cscdqm {

  class EventProcessorMutex : public Lock {

    private:

      EventProcessor processor;
      Configuration *config;

    public:

      EventProcessorMutex(Configuration* const p_config) : processor(p_config) {
        config = p_config;
      }
      void updateFractionAndEfficiencyHistos();

  };

  static const std::type_info& EMUHistoDefT = typeid(cscdqm::EMUHistoDef);
  static const std::type_info& DDUHistoDefT = typeid(cscdqm::DDUHistoDef);
  static const std::type_info& CSCHistoDefT = typeid(cscdqm::CSCHistoDef);
  static const std::type_info& ParHistoDefT = typeid(cscdqm::ParHistoDef);

  /** CSC Id structure */
  typedef struct CSCHwId {

    HwId crateId;
    HwId dmbId;

    CSCHwId(const HistoDef& hdef) : crateId(hdef.getCrateId()), dmbId(hdef.getDMBId()) { }

    const CSCHwId& operator= (const CSCHwId& id) {
      crateId = id.crateId;
      dmbId   = id.dmbId;
      return *this;
    }

    const bool operator== (const CSCHwId& id) const {
      return (crateId == id.crateId && dmbId == id.dmbId);
    }

    const bool operator< (const CSCHwId& id) const {
      if (crateId < id.crateId)  return true;
      if (dmbId < id.dmbId) return true;
      return false;
    }

    friend std::ostream& operator<<(std::ostream& out, const CSCHwId& id) {
      return out << "CSC:" << id.crateId << ":" << id.dmbId;
    }

  };

  typedef boost::multi_index_container<
    CSCHwId,
    boost::multi_index::indexed_by<
      boost::multi_index::ordered_unique< 
        boost::multi_index::composite_key<
          CSCHwId,
          boost::multi_index::member<CSCHwId, HwId, &CSCHwId::crateId>,
          boost::multi_index::member<CSCHwId, HwId, &CSCHwId::dmbId>
          >
        >
      >
    > CSCHwIdSet;

  /**
   * @class Dispatcher
   * @brief Framework frontend and Histogram Cache controller
   */
  class Dispatcher {

    public:

      Dispatcher(Configuration* const p_config, MonitorObjectProvider* const p_provider);
      ~Dispatcher() { threads.join_all(); }

      void init();

      void updateFractionAndEfficiencyHistos();
      const bool nextBookedCSC(unsigned int& n, unsigned int& crateId, unsigned int& dmbId) const;

      Collection* getCollection() { return &collection; }
       
      const bool getHisto(const HistoDef& histoD, MonitorObject*& me);

    private:

      void updateFractionAndEfficiencyHistosAuto();

      Configuration         *config;
      MonitorObjectProvider *provider;
      Collection            collection;
      EventProcessor        processor;
      Cache                 cache;

      EventProcessorMutex processorFract;
      boost::thread_group threads;
      boost::function<void ()> fnUpdate;
      std::set<HwId> bookedDDUs;
      CSCHwIdSet bookedCSCs;

#ifdef DQMLOCAL

    public:

      void processEvent(const char* data, const int32_t dataSize, const uint32_t errorStat, const int32_t nodeNumber);

#endif      

#ifdef DQMGLOBAL

    public:

      void processEvent(const edm::Event& e, const edm::InputTag& inputTag);

#endif      

  };

}

#endif
