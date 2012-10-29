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

#ifdef DQMMT      
#include <boost/thread.hpp>
#endif

#include "CSCDQM_Configuration.h"
#include "CSCDQM_EventProcessor.h"
#include "CSCDQM_Collection.h"
#include "CSCDQM_Cache.h"
#include "CSCDQM_Logger.h"
#include "CSCDQM_Lock.h"

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

      /** If full standby was already processed? */
      bool fullStandbyProcessed;

      /** Last standby value. To be checked for HV changes */
      HWStandbyType lastStandby;

    public:

      /**
       * @brief  Constructor.
       * @param  p_config Pointer to Global Configuration
       */
      EventProcessorMutex(Configuration* const p_config) : processor(p_config) {
        config = p_config;
        fullStandbyProcessed = false;
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

      /**
        * @brief  Mask HW elements from the efficiency calculations. Can be applied on runtime!
        * @param  tokens String tokens of the HW elements
        * @return elements masked
        */
      unsigned int maskHWElements(std::vector<std::string>& tokens) {
        return processor.maskHWElements(tokens);
      }

      
      /**
       * @brief  Process standby information
       * @param  standby Standby information
       */
      void processStandby(HWStandbyType& standby) {
        if (lastStandby != standby) {
          processor.standbyEfficiencyHistos(standby);
          if (config->getIN_FULL_STANDBY()) {
            // Lets mark CSCs as BAD - have not ever ever been in !STANDBY 
            if (!fullStandbyProcessed) {
              processor.standbyEfficiencyHistos(standby);
              processor.writeShifterHistograms();
              fullStandbyProcessed = true;
            }
          }
          lastStandby = standby;
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
      ~Dispatcher() { 
#ifdef DQMMT      
        threads.join_all(); 
#endif      
      }

      void init();
      void updateFractionAndEfficiencyHistos();
      const bool getHisto(const HistoDef& histoD, MonitorObject*& me);
      unsigned int maskHWElements(std::vector<std::string>& tokens);
      void processStandby(HWStandbyType& standby);

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

#ifdef DQMMT      

      /** Thread group to store all threads created by Dispatcher */
      boost::thread_group threads;

#endif      

#ifdef DQMLOCAL

    public:

      void processEvent(const char* data, const int32_t dataSize, const uint32_t errorStat, const int32_t nodeNumber);

#endif      

#ifdef DQMGLOBAL

    public:

      void processEvent(const edm::Event& e, const edm::InputTag& inputTag, HWStandbyType& standby);

#endif      

  };

}

#endif
