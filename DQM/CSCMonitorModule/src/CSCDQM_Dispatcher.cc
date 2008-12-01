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
  }

  const bool Dispatcher::getHisto(const HistoType& histoT, MonitorObject*& me) {
    if (cache.get(histoT, me)) return true;
    bool ret = config->provider->getHisto(histoT, me);
    cache.put(histoT, me);
    return ret;
  }

  void Dispatcher::updateFractionAndEfficiencyHistos() {
    boost::function<void ()> fnUpdate = boost::bind(&EventProcessorMutex::updateFractionAndEfficiencyHistos, &processorFract);
    boost::thread(boost::ref(fnUpdate));
  }


}
