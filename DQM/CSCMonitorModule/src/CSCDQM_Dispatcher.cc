/*
 * =====================================================================================
 *
 *       Filename:  CSCDQM_Dispatcher.cc
 *
 *    Description:  CSCDQM Dispatcher implementation
 *
 *        Version:  1.0
 *        Created:  12/01/2008 10:32:38 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius (VR), valdas.rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */

#include "DQM/CSCMonitorModule/interface/CSCDQM_Dispatcher.h"

namespace cscdqm {

  Dispatcher::Dispatcher(Configuration* const p_config) : collection(p_config), processor(p_config), processorFract(p_config) {
    config = p_config;
    config->getHisto = boost::bind(&Dispatcher::getHisto, this, _1, _2);
    fnUpdate = boost::bind(&EventProcessorMutex::updateFractionAndEfficiencyHistos, &processorFract);
  }

  const bool Dispatcher::getHisto(const HistoType& histoT, MonitorObject*& me) {
    if (cache.get(histoT, me)) return true;
    bool ret = config->provider->getHisto(histoT, me);
    cache.put(histoT, me);
    return ret;
  }

  void Dispatcher::updateFractionAndEfficiencyHistosAuto() {
    if ( config->FRAEFF_AUTO_UPDATE &&
        (config->getNEventsCSC() > config->FRAEFF_AUTO_UPDATE_START) &&
        (config->getNEventsCSC() % config->FRAEFF_AUTO_UPDATE_FREQ) == 0) {
      updateFractionAndEfficiencyHistos();
    }
  }

  void Dispatcher::updateFractionAndEfficiencyHistos() {
    if (!processorFract.isLocked()) {
      processorFract.lock();
      if (config->FRAEFF_SEPARATE_THREAD) { 
        threads.create_thread(boost::ref(fnUpdate));
        threads.join_all();
      } else {
        fnUpdate();
      }
      processorFract.unlock();
    }
  }

#ifdef DQMLOCAL

  void Dispatcher::processEvent(const char* data, const int32_t dataSize, const uint32_t errorStat, const int32_t nodeNumber) {
    processor.processEvent(data, dataSize, errorStat, nodeNumber);
    updateFractionAndEfficiencyHistosAuto();
  }

#endif      

#ifdef DQMGLOBAL

  void Dispatcher::processEvent(const edm::Event& e, const edm::InputTag& inputTag) {
    processor.processEvent(e, inputTag);
    updateFractionAndEfficiencyHistosAuto();
  }

#endif      

}
