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

#include "DQM/CSCMonitorModule/interface/CSCDQM_Configuration.h"
#include "DQM/CSCMonitorModule/interface/CSCDQM_EventProcessor.h"
#include "DQM/CSCMonitorModule/interface/CSCDQM_Collection.h"

namespace cscdqm {

  /**
   * @class Dispatcher
   * @brief Framework frontend and Histogram Cache controller
   */
  class Dispatcher {

    public:

      Dispatcher(Configuration* const p_config) : collection(p_config), processor(p_config) {
        config = p_config;
      }

      void updateFractionHistos() { processor.updateFractionHistos(); }
      void updateEfficiencyHistos() { processor.updateEfficiencyHistos(); }

      Collection* getCollection() { return &collection; }
       
    private:

      cscdqm::Configuration  *config;
      cscdqm::Collection     collection;
      cscdqm::EventProcessor processor;
    
// ===================================================================================================
// Local ONLY stuff 
// ===================================================================================================

#ifdef DQMLOCAL

    public:

      void processEvent(const char* data, const int32_t dataSize, const uint32_t errorStat, const int32_t nodeNumber) {
        processor.processEvent(data, dataSize, errorStat, nodeNumber);
      }

#endif      

// ===================================================================================================
// Global ONLY stuff 
// ===================================================================================================

#ifdef DQMGLOBAL

    public:

      void processEvent(const edm::Event& e, const edm::InputTag& inputTag) {
        processor.processEvent(e, inputTag);
      }

#endif      

  };

}

#endif
