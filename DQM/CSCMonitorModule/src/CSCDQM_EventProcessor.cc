/*
 * =====================================================================================
 *
 *       Filename:  EventProcessor.cc
 *
 *    Description:  EventProcessor Object
 *
 *        Version:  1.0
 *        Created:  10/03/2008 10:47:11 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius (VR), Valdas.Rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */

#include "DQM/CSCMonitorModule/interface/CSCDQM_EventProcessor.h"

namespace cscdqm {


  EventProcessor::EventProcessor(Configuration* const p_config) {

    config = p_config;
    fFirstEvent = true;
    fCloseL1As = true;

  }

  void EventProcessor::init() {

    binChecker.crcALCT(config->getBINCHECKER_CRC_ALCT());
    binChecker.crcTMB(config->getBINCHECKER_CRC_CLCT());
    binChecker.crcCFEB(config->getBINCHECKER_CRC_CFEB());
    binChecker.modeDDU(config->getBINCHECKER_MODE_DDU());

    if (config->getBINCHECKER_OUTPUT()) {
      binChecker.output1().show();
      binChecker.output2().show();
    } else {
      binChecker.output1().hide();
      binChecker.output2().hide();
    }

  }

  const bool EventProcessor::getEMUHisto(const HistoId& histo, MonitorObject*& me) {
    if (config->fnGetCacheEMUHisto(histo, me)) return (me != NULL);
    EMUHistoDef histoD(histo);
    if (config->fnGetHisto(histoD, me)) return (me != NULL);
    return false;
  }


  const bool EventProcessor::getDDUHisto(const HistoId& histo, const HwId& dduID, MonitorObject*& me) {
    if (config->fnGetCacheDDUHisto(histo, dduID, me)) return (me != NULL);
    DDUHistoDef histoD(histo, dduID);
    if (config->fnGetHisto(histoD, me)) return (me != NULL);
    return false;
  }


  const bool EventProcessor::getCSCHisto(const HistoId& histo, const HwId& crateID, const HwId& dmbSlot, MonitorObject*& me) {
    if (config->fnGetCacheCSCHisto(histo, crateID, dmbSlot, 0, me)) return (me != NULL);
    CSCHistoDef histoD(histo, crateID, dmbSlot);
    if (config->fnGetHisto(histoD, me)) return (me != NULL);
    return false;
  }


  const bool EventProcessor::getCSCHisto(const HistoId& histo, const HwId& crateID, const HwId& dmbSlot, const HwId& adId, MonitorObject*& me) {
    if (config->fnGetCacheCSCHisto(histo, crateID, dmbSlot, adId, me)) return (me != NULL);
    CSCHistoDef histoD(histo, crateID, dmbSlot, adId);
    if (config->fnGetHisto(histoD, me)) return (me != NULL);
    return false;
  }


  const bool EventProcessor::getParHisto(const HistoId& histo, MonitorObject*& me) {
    if (config->fnGetCacheParHisto(histo, me)) return (me != NULL);
    ParHistoDef histoD(histo);
    if (config->fnGetHisto(histoD, me)) return (me != NULL);
    return false;
  }

  const bool EventProcessor::getCSCFromMap(const unsigned int& crateId, const unsigned int& dmbId, unsigned int& cscType, unsigned int& cscPosition) const {

    if (crateId < 1 || crateId > 60 || dmbId < 1 || dmbId > 10) return false;

    CSCDetId cid = config->fnGetCSCDetId(crateId, dmbId);
    cscPosition  = cid.chamber();
    int iring    = cid.ring();
    int istation = cid.station();
    int iendcap  = cid.endcap();
    std::string tlabel = cscdqm::Utility::getCSCTypeLabel(iendcap, istation, iring);
    cscType = cscdqm::Utility::getCSCTypeBin(tlabel);

    return true;

  }

}
