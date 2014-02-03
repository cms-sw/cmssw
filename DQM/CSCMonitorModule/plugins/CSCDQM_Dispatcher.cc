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

#include "CSCDQM_Dispatcher.h"

namespace cscdqm {

  /**
   * @brief  Constructor.
   * @param  p_config Pointer to Global Configuration
   * @param  p_provider Pointer to MonitorObjectProvider
   * @return 
   */
  Dispatcher::Dispatcher(Configuration* const p_config, MonitorObjectProvider* p_provider) : 
  collection(p_config), processor(p_config), processorFract(p_config) {

    /** Save pointers to class properties */
    config = p_config;
    provider = p_provider;

    /** Link/share Cache methods to function pointers in configuration */
    config->fnGetCacheEMUHisto = boost::bind(&Cache::getEMU, &cache, _1, _2);
    config->fnGetCacheFEDHisto = boost::bind(&Cache::getFED, &cache, _1, _2, _3);
    config->fnGetCacheDDUHisto = boost::bind(&Cache::getDDU, &cache, _1, _2, _3);
    config->fnGetCacheCSCHisto = boost::bind(&Cache::getCSC, &cache, _1, _2, _3, _4, _5);
    config->fnGetCacheParHisto = boost::bind(&Cache::getPar, &cache, _1, _2);
    config->fnPutHisto = boost::bind(&Cache::put, &cache, _1, _2);
    config->fnNextBookedCSC = boost::bind(&Cache::nextBookedCSC, &cache, _1, _2, _3);
    config->fnIsBookedCSC = boost::bind(&Cache::isBookedCSC, &cache, _1, _2);
    config->fnIsBookedDDU = boost::bind(&Cache::isBookedDDU, &cache, _1);
    config->fnIsBookedFED = boost::bind(&Cache::isBookedFED, &cache, _1);

    /** Link/share local functions */
    config->fnGetHisto = boost::bind(&Dispatcher::getHisto, this, _1, _2);

    /** Link/share getCSCDetId function */
    config->fnGetCSCDetId = boost::bind(&MonitorObjectProvider::getCSCDetId, provider, _1, _2, _3);

    /** Link/share booking function */
    config->fnBook = boost::bind(&MonitorObjectProvider::bookMonitorObject, provider, _1);

  }

  /**
   * @brief  Initialize Dispatcher: book histograms, init processor, etc.
   * @return 
   */
  void Dispatcher::init() {
    collection.load();
    collection.bookEMUHistos();
    processor.init();
  }

  /**
   * @brief  Mask HW elements from the efficiency calculations. Can be applied on runtime!
   * @param  tokens String tokens of the HW elements
   * @return elements masked
   */
  unsigned int Dispatcher::maskHWElements(std::vector<std::string>& tokens) {
    return processorFract.maskHWElements(tokens);
  }

  /**
   * @brief  Global get MO function. If request has reached this function it means that histo is not in cache!
   * @param  histoD Histogram Definition to get
   * @param  me MO to return
   * @return true if me found and filled, false - otherwise
   */
  const bool Dispatcher::getHisto(const HistoDef& histoD, MonitorObject*& me) {

    /** For the first FED - book general */
    if (typeid(histoD) == FEDHistoDefT && !cache.isBookedFED(histoD.getFEDId())) {
      collection.bookFEDHistos(histoD.getFEDId());
      if (cache.get(histoD, me)) return true;
    }

    /** For the first DDU - book general */
    if (typeid(histoD) == DDUHistoDefT && !cache.isBookedDDU(histoD.getDDUId())) {
      collection.bookDDUHistos(histoD.getDDUId());
      if (cache.get(histoD, me)) return true;
    }

    /** For the first and specific CSCs - book general and specific */
    if (typeid(histoD) == CSCHistoDefT) {
      if (!cache.isBookedCSC(histoD.getCrateId(), histoD.getDMBId())) {
        collection.bookCSCHistos(histoD.getCrateId(), histoD.getDMBId());
        /** cache.printContent(); */
      }
      if (collection.isOnDemand(histoD.getHistoName())) {
        collection.bookCSCHistos(histoD.getId(), histoD.getCrateId(), histoD.getDMBId(), histoD.getAddId());
      }
      if (cache.get(histoD, me)) return true;
    }

    /** For the Parameters - book parameter histogram */
    if (typeid(histoD) == ParHistoDefT) {
      HistoBookRequest req(histoD, config->getFOLDER_PAR(), -1.0f);
      me = provider->bookMonitorObject(req);
      cache.put(histoD, me);
      return true;
    }

    /** If not found after booking - mark it as not existent */
    cache.put(histoD, NULL);

    return false;
  }

  /**
   * @brief  Automatically called fraction and efficiency MOs update function
   * @return 
   */
  void Dispatcher::updateFractionAndEfficiencyHistosAuto() {
    if ( config->getFRAEFF_AUTO_UPDATE() &&
        (config->getNEventsCSC() >= config->getFRAEFF_AUTO_UPDATE_START()) &&
        (config->getNEventsCSC() % config->getFRAEFF_AUTO_UPDATE_FREQ()) == 0) {
      updateFractionAndEfficiencyHistos();
    }
  }

  /**
   * @brief  On demand update fraction and efficiency MOs
   * @return 
   */
  void Dispatcher::updateFractionAndEfficiencyHistos() {
    LockType lock(processorFract.mutex);
    if (config->getFRAEFF_SEPARATE_THREAD()) { 
      boost::function<void ()> fnUpdate = boost::bind(&EventProcessorMutex::updateFractionAndEfficiencyHistos, &processorFract);
#ifdef DQMMT
      threads.create_thread(boost::ref(fnUpdate));
#else
      fnUpdate();
#endif 
    } else {
      processorFract.updateFractionAndEfficiencyHistos();
    }
  }

  /**
   * @brief  Set HW Standby modes
   * @return 
   */
  void Dispatcher::processStandby(HWStandbyType& standby) {
    LockType lock(processorFract.mutex);
    if (config->getFRAEFF_SEPARATE_THREAD()) { 
      boost::function<void (HWStandbyType&)> fnUpdate = boost::bind(&EventProcessorMutex::processStandby, &processorFract, _1);
#ifdef DQMMT
      threads.create_thread(boost::ref(fnUpdate));
#else
      fnUpdate(standby);
#endif 
    } else {
      processorFract.processStandby(standby);
    }
  }

#ifdef DQMLOCAL

  /**
   * @brief  Process event (Local DQM)
   * @param  data Event Data buffer
   * @param  dataSize Event Data buffer size
   * @param  errorStat Error status received by reading DAQ buffer
   * @param  nodeNumber DAQ node number
   * @return 
   */
  void Dispatcher::processEvent(const char* data, const int32_t dataSize, const uint32_t errorStat, const int32_t nodeNumber) {
    config->eventProcessTimer(true);
    processor.processEvent(data, dataSize, errorStat, nodeNumber);
    config->eventProcessTimer(false);
    updateFractionAndEfficiencyHistosAuto();
  }

#endif      

#ifdef DQMGLOBAL

  /**
   * @brief  Process event (Global DQM)
   * @param  e Event object
   * @param  inputTag Tag to search Event Data in
   * @return 
   */
  void Dispatcher::processEvent(const edm::Event& e, const edm::InputTag& inputTag, HWStandbyType& standby) {
    config->eventProcessTimer(true);
     
    // Consider standby information
    if (standby.process) {

      // Set in full standby once at the start. Afterwards - no!
      // i.e. if we ever in the run have gone off standby - this value is false 
      config->setIN_FULL_STANDBY(config->getIN_FULL_STANDBY() && standby.fullStandby());

      //std::cout << "standby.MeP = " << standby.MeP << "\n";
      //std::cout << "standby.MeM = " << standby.MeM << "\n";
      //std::cout << "standby.fullStandby() = " << standby.fullStandby() << "\n";
      //std::cout << "config->getIN_FULL_STANDBY = " << config->getIN_FULL_STANDBY() << "\n";

      processStandby(standby);

      // We do not fill histograms in full standby!
      if (standby.fullStandby()) {
        return;
      }

    }

    processor.processEvent(e, inputTag);

    config->eventProcessTimer(false);

    updateFractionAndEfficiencyHistosAuto();

  }

#endif      

}
