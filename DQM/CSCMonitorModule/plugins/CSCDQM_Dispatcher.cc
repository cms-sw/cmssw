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
  Dispatcher::Dispatcher(Configuration* const p_config, MonitorObjectProvider* p_provider)
      : collection(p_config), processor(p_config), processorFract(p_config) {
    commonConstruct(p_config, p_provider);
  }

#ifdef DQMGLOBAL

  /**
 * @brief  Constructor.
 * @param  p_config Pointer to Global Configuration
 * @param  p_provider Pointer to MonitorObjectProvider
 * @param  itag InputTag to raw data - to be passed down
 * @param  coco rvalue to ConsumesCollector - to be passed down
 * @return
 */
  Dispatcher::Dispatcher(Configuration* const p_config,
                         MonitorObjectProvider* p_provider,
                         const edm::InputTag& itag,
                         edm::ConsumesCollector&& coco)
      : collection(p_config), processor(p_config, itag, coco), processorFract(p_config) {
    commonConstruct(p_config, p_provider);
  }

#endif

  void Dispatcher::commonConstruct(Configuration* const p_config, MonitorObjectProvider* p_provider) {
    /** Save pointers to class properties */
    config = p_config;
    provider = p_provider;

    /** Link/share Cache methods to function pointers in configuration */
    config->fnGetCacheEMUHisto = std::bind(&Cache::getEMU, &cache, std::placeholders::_1, std::placeholders::_2);
    config->fnGetCacheFEDHisto =
        std::bind(&Cache::getFED, &cache, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
    config->fnGetCacheDDUHisto =
        std::bind(&Cache::getDDU, &cache, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
    config->fnGetCacheCSCHisto = std::bind(&Cache::getCSC,
                                           &cache,
                                           std::placeholders::_1,
                                           std::placeholders::_2,
                                           std::placeholders::_3,
                                           std::placeholders::_4,
                                           std::placeholders::_5);
    config->fnGetCacheParHisto = std::bind(&Cache::getPar, &cache, std::placeholders::_1, std::placeholders::_2);
    config->fnPutHisto = std::bind(&Cache::put, &cache, std::placeholders::_1, std::placeholders::_2);
    config->fnNextBookedCSC =
        std::bind(&Cache::nextBookedCSC, &cache, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
    config->fnIsBookedCSC = std::bind(&Cache::isBookedCSC, &cache, std::placeholders::_1, std::placeholders::_2);
    config->fnIsBookedDDU = std::bind(&Cache::isBookedDDU, &cache, std::placeholders::_1);
    config->fnIsBookedFED = std::bind(&Cache::isBookedFED, &cache, std::placeholders::_1);

    /** Link/share local functions */
    config->fnGetHisto = std::bind(&Dispatcher::getHisto, this, std::placeholders::_1, std::placeholders::_2);

    /** Link/share getCSCDetId function */
    config->fnGetCSCDetId = std::bind(&MonitorObjectProvider::getCSCDetId,
                                      provider,
                                      std::placeholders::_1,
                                      std::placeholders::_2,
                                      std::placeholders::_3);

    /** Link/share booking function */
    config->fnBook = std::bind(&MonitorObjectProvider::bookMonitorObject, provider, std::placeholders::_1);
  }

  /**
 * @brief  Initialize Dispatcher: book histograms, init processor, etc.
 * @return
 */
  void Dispatcher::init() {
    collection.load();
    // collection.bookEMUHistos();
    // processor.init();
  }

  void Dispatcher::book() {
    collection.bookEMUHistos();

    /*** FOr multi-threading pre-book all FED, DDU, CSC histograms ***/
    if (config->getPREBOOK_ALL_HISTOS()) {
      /** For the first FED - book general */
      for (HwId FEDId = 750; FEDId < 758; FEDId++) {
        if (!cache.isBookedFED(FEDId)) {
          collection.bookFEDHistos(FEDId);
        }
      }

      if (config->getPROCESS_DDU()) {
        /** For the first DDU - book general */
        for (HwId DDUId = 1; DDUId <= 36; DDUId++) {
          if (!cache.isBookedDDU(DDUId)) {
            collection.bookDDUHistos(DDUId);
          }
        }
      }

      if (config->getPROCESS_CSC()) {
        /** For the first and specific CSCs - book general and specific */
        for (HwId CrateId = 1; CrateId <= 60; CrateId++) {
          for (HwId DMBId = 1; DMBId <= 10; DMBId++) {
            if (DMBId == 6)
              continue;  /// No DMB in slot 6
            if (!cache.isBookedCSC(CrateId, DMBId)) {
              collection.bookCSCHistos(CrateId, DMBId);
            }
          }
        }
      }

      if (config->getPROCESS_EFF_PARAMETERS()) {
        /** For the Parameters - book parameter histograms */

        std::vector<HistoId> parameters;
        parameters.push_back(h::PAR_CSC_SIDEPLUS_STATION01_RING01);
        parameters.push_back(h::PAR_CSC_SIDEPLUS_STATION01_RING02);
        parameters.push_back(h::PAR_CSC_SIDEPLUS_STATION01_RING03);
        parameters.push_back(h::PAR_CSC_SIDEPLUS_STATION01);
        parameters.push_back(h::PAR_CSC_SIDEPLUS_STATION02_RING01);
        parameters.push_back(h::PAR_CSC_SIDEPLUS_STATION02_RING02);
        parameters.push_back(h::PAR_CSC_SIDEPLUS_STATION02);
        parameters.push_back(h::PAR_CSC_SIDEPLUS_STATION03_RING01);
        parameters.push_back(h::PAR_CSC_SIDEPLUS_STATION03_RING02);
        parameters.push_back(h::PAR_CSC_SIDEPLUS_STATION03);
        parameters.push_back(h::PAR_CSC_SIDEPLUS_STATION04_RING01);
        parameters.push_back(h::PAR_CSC_SIDEPLUS_STATION04_RING02);
        parameters.push_back(h::PAR_CSC_SIDEPLUS_STATION04);
        parameters.push_back(h::PAR_CSC_SIDEPLUS);
        parameters.push_back(h::PAR_CSC_SIDEMINUS_STATION01_RING01);
        parameters.push_back(h::PAR_CSC_SIDEMINUS_STATION01_RING02);
        parameters.push_back(h::PAR_CSC_SIDEMINUS_STATION01_RING03);
        parameters.push_back(h::PAR_CSC_SIDEMINUS_STATION01);
        parameters.push_back(h::PAR_CSC_SIDEMINUS_STATION02_RING01);
        parameters.push_back(h::PAR_CSC_SIDEMINUS_STATION02_RING02);
        parameters.push_back(h::PAR_CSC_SIDEMINUS_STATION02);
        parameters.push_back(h::PAR_CSC_SIDEMINUS_STATION03_RING01);
        parameters.push_back(h::PAR_CSC_SIDEMINUS_STATION03_RING02);
        parameters.push_back(h::PAR_CSC_SIDEMINUS_STATION03);
        parameters.push_back(h::PAR_CSC_SIDEMINUS_STATION04_RING01);
        parameters.push_back(h::PAR_CSC_SIDEMINUS_STATION04_RING02);
        parameters.push_back(h::PAR_CSC_SIDEMINUS_STATION04);
        parameters.push_back(h::PAR_CSC_SIDEMINUS);
        parameters.push_back(h::PAR_REPORT_SUMMARY);

        for (size_t i = 0; i < parameters.size(); i++) {
          ParHistoDef histoD(parameters[i]);
          auto histodef = HistoDef(parameters[i]);
          HistoBookRequest req(histodef, config->getFOLDER_PAR(), -1.0f);
          MonitorObject* me = provider->bookMonitorObject(req);
          cache.put(histoD, me);
        }
      }
    }

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
      if (cache.get(histoD, me))
        return true;
    }

    /** For the first DDU - book general */
    if (typeid(histoD) == DDUHistoDefT && !cache.isBookedDDU(histoD.getDDUId())) {
      collection.bookDDUHistos(histoD.getDDUId());
      if (cache.get(histoD, me))
        return true;
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
      if (cache.get(histoD, me))
        return true;
    }

    /** For the Parameters - book parameter histogram */
    if (typeid(histoD) == ParHistoDefT) {
      HistoBookRequest req(histoD, config->getFOLDER_PAR(), -1.0f);
      me = provider->bookMonitorObject(req);
      cache.put(histoD, me);
      return true;
    }

    /** If not found after booking - mark it as not existent */
    cache.put(histoD, nullptr);

    return false;
  }

  /**
 * @brief  Automatically called fraction and efficiency MOs update function
 * @return
 */
  void Dispatcher::updateFractionAndEfficiencyHistosAuto() {
    if (config->getFRAEFF_AUTO_UPDATE() && (config->getNEventsCSC() >= config->getFRAEFF_AUTO_UPDATE_START()) &&
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
      std::function<void()> fnUpdate =
          std::bind(&EventProcessorMutex::updateFractionAndEfficiencyHistos, &processorFract);
#ifdef DQMMT
      threads.create_thread(std::ref(fnUpdate));
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
      std::function<void(HWStandbyType&)> fnUpdate =
          std::bind(&EventProcessorMutex::processStandby, &processorFract, std::placeholders::_1);
#ifdef DQMMT
      threads.create_thread(std::ref(fnUpdate));
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
  void Dispatcher::processEvent(const char* data,
                                const int32_t dataSize,
                                const uint32_t errorStat,
                                const int32_t nodeNumber) {
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

}  // namespace cscdqm
