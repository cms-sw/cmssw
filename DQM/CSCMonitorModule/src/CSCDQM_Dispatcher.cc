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

  Dispatcher::Dispatcher(Configuration* const p_config, MonitorObjectProvider* p_provider) : 
  collection(p_config), processor(p_config), processorFract(p_config) {

    config = p_config;
    provider = p_provider;

    config->fnGetHisto = boost::bind(&Dispatcher::getHisto, this, _1, _2);
    config->fnGetCacheHisto = boost::bind(&Cache::get, &cache, _1, _2, _3, _4, _5, _6);
    config->fnPutHisto = boost::bind(&Cache::put, &cache, _1, _2);
    config->fnBook = boost::bind(&MonitorObjectProvider::bookMonitorObject, provider, _1);
    config->fnGetCSCDetId = boost::bind(&MonitorObjectProvider::getCSCDetId, provider, _1, _2);
    config->fnNextBookedCSC = boost::bind(&Dispatcher::nextBookedCSC, this, _1, _2, _3);

    fnUpdate = boost::bind(&EventProcessorMutex::updateFractionAndEfficiencyHistos, &processorFract);

  }

  void Dispatcher::init() {
    collection.bookEMUHistos();
    processor.init();
  }

  void EventProcessorMutex::updateFractionAndEfficiencyHistos() {
    lock();

    config->updateFraTimer(true);
    processor.updateFractionHistos();
    config->updateFraTimer(false);

    if (config->getPROCESS_EFF_HISTOS()) {

      config->updateEffTimer(true);
      processor.updateEfficiencyHistos();
      config->updateEffTimer(false);

    }
    unlock();
  }

  const bool Dispatcher::getHisto(const HistoDef& histoD, MonitorObject*& me) {

    //Look at the cache - if found - return it 
    //if (cache.get(histoD, me)) return true;

    //LOG_DEBUG << "DISPATCHER: need to book histo on " << histoD;

    //For the first DDU - book general
    if (typeid(histoD) == DDUHistoDefT && bookedDDUs.find(histoD.getDDUId()) == bookedDDUs.end()) {
      collection.bookDDUHistos(histoD.getDDUId());
      bookedDDUs.insert(histoD.getDDUId());
      if (cache.get(histoD, me)) return true;
    }

    //For the first and specific CSCs - book general and specific
    if (typeid(histoD) == CSCHistoDefT) {
      CSCHwId cscId(histoD);
      //LOG_DEBUG << "DISPATCHER: looking for " << cscId;
      if (bookedCSCs.find(boost::make_tuple(histoD.getCrateId(), histoD.getDMBId())) == bookedCSCs.end()) {
        collection.bookCSCHistos(histoD.getCrateId(), histoD.getDMBId());
        //LOG_DEBUG << "DISPATCHER: booked histos for " << cscId;
        bookedCSCs.insert(cscId);
        //cache.printContent();
      }
      if (collection.isOnDemand(histoD.getHistoName())) {
        //LOG_DEBUG << "DISPATCHER: booking on demand " << histoD;
        collection.bookCSCHistos(histoD.getId(), histoD.getCrateId(), histoD.getDMBId(), histoD.getAddId());
      }
      if (cache.get(histoD, me)) return true;
    }

    //For the Parameters - book parameter histogram
    if (typeid(histoD) == ParHistoDefT) {
      me = provider->bookMonitorObject(HistoBookRequest(histoD, config->getFOLDER_PAR(), -1.0f));
      cache.put(histoD, me);
      return true;
    }

    //If not found after booking - mark it as not existent
    cache.put(histoD, NULL);

    return false;
  }

  void Dispatcher::updateFractionAndEfficiencyHistosAuto() {
    if ( config->getFRAEFF_AUTO_UPDATE() &&
        (config->getNEventsCSC() >= config->getFRAEFF_AUTO_UPDATE_START()) &&
        (config->getNEventsCSC() % config->getFRAEFF_AUTO_UPDATE_FREQ()) == 0) {
      updateFractionAndEfficiencyHistos();
    }
  }

  void Dispatcher::updateFractionAndEfficiencyHistos() {
    if (!processorFract.isLockedByOther()) {
      if (config->getFRAEFF_SEPARATE_THREAD()) { 
        threads.create_thread(boost::ref(fnUpdate));
      } else {
        fnUpdate();
      }
    }
  }

  const bool Dispatcher::nextBookedCSC(unsigned int& n, unsigned int& crateId, unsigned int& dmbId) const {
    if (n < bookedCSCs.size()) {
      CSCHwIdSet::iterator iter = bookedCSCs.begin();
      for (unsigned int i = n; i > 0; i--) iter++;
      crateId = iter->crateId;
      dmbId   = iter->dmbId;
      n++;
      return true;
    }
    return false;
  }

#ifdef DQMLOCAL

  void Dispatcher::processEvent(const char* data, const int32_t dataSize, const uint32_t errorStat, const int32_t nodeNumber) {
    config->eventProcessTimer(true);
    processor.processEvent(data, dataSize, errorStat, nodeNumber);
    config->eventProcessTimer(false);
    updateFractionAndEfficiencyHistosAuto();
  }

#endif      

#ifdef DQMGLOBAL

  void Dispatcher::processEvent(const edm::Event& e, const edm::InputTag& inputTag) {
    config->eventProcessTimer(true);
    processor.processEvent(e, inputTag);
    config->eventProcessTimer(false);
    updateFractionAndEfficiencyHistosAuto();
  }

#endif      

}
