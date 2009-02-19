/*
 * =====================================================================================
 *
 *       Filename:  CSCDQM_Dispatcher.h
 *
 *    Description:  CSCDQM Framework frontend and Histogram Cache controller
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

#include "DQM/CSCMonitorModule/interface/CSCDQM_Configuration.h"
#include "DQM/CSCMonitorModule/interface/CSCDQM_EventProcessor.h"
#include "DQM/CSCMonitorModule/interface/CSCDQM_Collection.h"
#include "DQM/CSCMonitorModule/interface/CSCDQM_Cache.h"
#include "DQM/CSCMonitorModule/interface/CSCDQM_Logger.h"
#include "DQM/CSCMonitorModule/interface/CSCDQM_Lock.h"

namespace cscdqm {

  /**
   * @class EventProcessorMutex
   * @brief Locking object (wrapper) that holds a separate EventProcessor. This
   * object can be used (theoretically) in separate thread.
   */
  class EventProcessorMutex : public Lock {

    private:

      /** Local (wrapped) event processor */
      EventProcessor processor;

      /** Global Configuration */
      Configuration *config;

    public:

      /**
       * @brief  Constructor.
       * @param  p_config Pointer to Global Configuration
       */
      EventProcessorMutex(Configuration* const p_config) : processor(p_config) {
        config = p_config;
      }

      /**
       * @brief  Update Fraction and Efficiency histograms
       * @return 
       */
      void updateFractionAndEfficiencyHistos() {
        LockType lock(mutex);
        config->updateFraTimer(true);
        processor.updateFractionHistos();
        config->updateFraTimer(false);
        if (config->getPROCESS_EFF_HISTOS()) {
          config->updateEffTimer(true);
          processor.updateEfficiencyHistos();
          config->updateEffTimer(false);
        }
      }

  };

  /**
   * @class Dispatcher
   * @brief CSCDQM Framework frontend and Histogram Cache controller
   */
  class Dispatcher {

    public:

      Dispatcher(Configuration* const p_config, MonitorObjectProvider* const p_provider);

      /**
       * @brief  Destructor. Joins and waits to complete all threads.
       */
      ~Dispatcher() { threads.join_all(); }

      void init();
      void updateFractionAndEfficiencyHistos();
      const bool getHisto(const HistoDef& histoD, MonitorObject*& me);

    private:

      void updateFractionAndEfficiencyHistosAuto();

      /** Pointer to Global Configuration */
      Configuration         *config;

      /** Pointer to MO provider */
      MonitorObjectProvider *provider;

      /** MO Collection object */
      Collection            collection;

      /** Event Processor object */
      EventProcessor        processor;

      /** MO Cache object */
      Cache                 cache;

      /** Lockable Fractional and Efficiency MO update object */
      EventProcessorMutex processorFract;

      /** Thread group to store all threads created by Dispatcher */
      boost::thread_group threads;

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
