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

  const bool EventProcessor::getEMUHisto(const HistoId& histo, MonitorObject*& me) {
    EMUHistoDef histoD(histo);
    if (config->fnGetHisto(histoD, me)) return (me != NULL);
    return false;
  }


  const bool EventProcessor::getDDUHisto(const HistoId& histo, const HwId dduID, MonitorObject*& me) {
    DDUHistoDef histoD(histo, dduID);
    if (config->fnGetHisto(histoD, me)) return (me != NULL);
    return false;
  }


  const bool EventProcessor::getCSCHisto(const HistoId& histo, const HwId crateID, const HwId dmbSlot, MonitorObject*& me) {
    CSCHistoDef histoD(histo, crateID, dmbSlot);
    if (config->fnGetHisto(histoD, me)) return (me != NULL);
    return false;
  }


  const bool EventProcessor::getCSCHisto(const HistoId& histo, const HwId crateID, const HwId dmbSlot, const HwId adId, MonitorObject*& me) {
    CSCHistoDef histoD(histo, crateID, dmbSlot, adId);
    if (config->fnGetHisto(histoD, me)) return (me != NULL);
    return false;
  }


  const bool EventProcessor::getParHisto(const HistoId& histo, MonitorObject*& me) {
    ParHistoDef histoD(histo);
    if (config->fnGetHisto(histoD, me)) return (me != NULL);
    return false;
  }


  const bool EventProcessor::getParHisto(const std::string& name, MonitorObject*& me) {
    const HistoName histo = const_cast<HistoName>(name.c_str());
    ParHistoDef histoD(histo);
    if (config->fnGetHisto(histoD, me)) return (me != NULL);
    return false;
  }

  void EventProcessor::getCSCFromMap(const unsigned int crateId, const unsigned int dmbId, unsigned int& cscType, unsigned int& cscPosition) const {
    CSCDetId cid = config->fnGetCSCDetId(crateId, dmbId);
    cscPosition  = cid.chamber();
    int iring    = cid.ring();
    int istation = cid.station();
    int iendcap  = cid.endcap();
    std::string tlabel = cscdqm::Utility::getCSCTypeLabel(iendcap, istation, iring);
    cscType = cscdqm::Utility::getCSCTypeBin(tlabel);
  }

}
